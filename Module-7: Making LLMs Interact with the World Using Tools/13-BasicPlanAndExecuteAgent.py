# Langchain Code to implement a Plan-And-Execute Agent that researches news and stores it into a local deeplake dataset, and can do a RAG on it to generate reports.
import os
import requests
from newspaper import Article
import time
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM as Ollama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.agents import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain_core.tools import Tool

print("****Loading Environment Variables....****")
load_dotenv(dotenv_path=".env")
google_cse_id=os.getenv("GOOGLE_CSE_ID")
google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = DeepLake(dataset_path="Databases/Module-7/13_BasicPlanAndExecuteAgent", embedding_function=embeddings)
print("****Environment Variables Loaded Successfully****")

llm=Ollama(model="mistral", base_url="http://localhost:11434")

def google_search_wrapper(tool_input: str):
    return GoogleSearchAPIWrapper(
        google_api_key=google_api_key,
        google_cse_id=google_cse_id
    ).run(tool_input)

google_search_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=google_search_wrapper
)


query = input("Enter your news search query: ").strip()

search = GoogleSearchAPIWrapper(
    google_api_key=google_api_key,
    google_cse_id=google_cse_id
)

# Fetch and store articles from the search results (handle network/timeouts gracefully)
try:
    results = search.results(query, num_results=5)
    print("Search Results:\n", results)
except Exception as e:
    print(f"Initial web search failed: {e}\nProceeding with local DB only.")
    results = []
urls = [result['link'] for result in results]
print("Article URLs:", urls)

docs = []
for url in urls:
    try:
        article = Article(url)
        article.download()
        article.parse()
        if article.text.strip():
            doc = Document(page_content=article.text, metadata={"source": url, "title": article.title})
            docs.append(doc)
    except Exception as e:
        print(f"Error fetching {url}: {e}")

if docs:
    vector_db.add_documents(docs)
    print(f"Added {len(docs)} articles to DeepLake!")
    print("Articles added:")
    for doc in docs:
        print(f"- {doc.metadata.get('title', 'No Title')} ({doc.metadata.get('source', 'No URL')})")
else:
    print("No articles were added to DeepLake.")

retriever = vector_db.as_retriever()
retriever.search_kwargs['k'] = 3  # Number of docs to retrieve per query


def retrieve_n_docs_tool(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    texts = [doc.page_content for doc in docs]
    return "\n---\n".join(texts)


private_doc_tool = Tool(
    name="Search Private Docs",
    func=retrieve_n_docs_tool,
    description="Useful for answering questions about current events using stored news articles."
)


def ingest_urls(urls):
    """Fetch articles from the list of URLs and add them to the vector DB.

    Returns number of documents added.
    """
    new_docs = []
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            if article.text.strip():
                doc = Document(page_content=article.text, metadata={"source": url, "title": article.title})
                new_docs.append(doc)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    if new_docs:
        vector_db.add_documents(new_docs)
        print(f"Added {len(new_docs)} articles to DeepLake!")
        for doc in new_docs:
            print(f"- {doc.metadata.get('title', 'No Title')} ({doc.metadata.get('source', 'No URL')})")
    return len(new_docs)

planner = load_chat_planner(Ollama(model="mistral", base_url="http://localhost:11434"))
executor = load_agent_executor(
    Ollama(model="mistral", base_url="http://localhost:11434"),
    tools=[google_search_tool, private_doc_tool],
    verbose=True
)
agent = PlanAndExecute(
    planner=planner,
    executor=executor,
    verbose=True,
    handle_parsing_errors=True
)

rag_query = input("Ask a question about the news articles you just added (or a new topic): ").strip()

# Determine how many relevant docs we already have and only fetch the remainder up to `target_docs`.
target_docs = 5
existing_docs = retriever.get_relevant_documents(rag_query) or []
existing_count = len(existing_docs)

if existing_count >= target_docs:
    print(f"Found {existing_count} relevant documents in the local DB — answering from local data.")
    # create an executor that only has access to the private docs
    executor_local = load_agent_executor(
        Ollama(model="mistral", base_url="http://localhost:11434"),
        tools=[private_doc_tool],
        verbose=True,
    )
    agent_local = PlanAndExecute(
        planner=planner,
        executor=executor_local,
        verbose=True,
        handle_parsing_errors=True,
    )
    response = agent_local.run(rag_query)
    print("\nAgent Response (local):\n", response)
else:
    need = target_docs - existing_count
    print(f"Only {existing_count} relevant documents found locally; need {need} more — searching web for up to {need} new articles.")
    # collect existing sources to avoid duplicate ingestion
    existing_sources = set(doc.metadata.get('source') for doc in existing_docs if doc.metadata.get('source'))

    # fetch more results than `need` to allow filtering duplicates, then trim
    num_search_results = max(need * 3, need)
    try:
        results = search.results(rag_query, num_results=num_search_results)
    except Exception as e:
        print(f"Search failed: {e}")
        results = []

    urls = [res.get('link') for res in results if res.get('link')]
    # filter out existing and keep unique
    urls_to_add = []
    seen = set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        if u in existing_sources:
            continue
        urls_to_add.append(u)
        if len(urls_to_add) >= need:
            break

    if not urls_to_add:
        print("No new URLs to ingest from search results. Proceeding to answer from available local docs (if any).")
    else:
        print("Fetching and ingesting URLs:", urls_to_add)
        added = ingest_urls(urls_to_add)
        if added == 0:
            print("No articles could be ingested from search results. Proceeding to answer from available local docs (if any).")
        else:
            # rebuild retriever reference to include new docs
            retriever = vector_db.as_retriever()
            retriever.search_kwargs['k'] = 3

    # answer using local-only agent (use whatever is in DB now)
    executor_local = load_agent_executor(
        Ollama(model="mistral", base_url="http://localhost:11434"),
        tools=[private_doc_tool],
        verbose=True,
    )
    agent_local = PlanAndExecute(
        planner=planner,
        executor=executor_local,
        verbose=True,
        handle_parsing_errors=True,
    )
    response = agent_local.run(rag_query)
    print("\nAgent Response:\n", response)