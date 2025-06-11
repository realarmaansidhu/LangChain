import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import DeepLake
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path="/.env")  # This loads variables from .env into the environment

print("Loaded API key:", os.getenv("GROQ_API_KEY"))

# instantiate the LLM and embeddings models
llm = Groq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.9
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

my_activeloop_org_id = "armaansidhu" 
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# create our documents
texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# add documents to our Deep Lake dataset
db.add_documents(docs)

retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm,
	chain_type="stuff",
	retriever=db.as_retriever()
)

# run the retrieval QA chain
response = retrieval_qa.invoke("When was Lady gaga born?")
print("Response:", response['result'])
response = retrieval_qa.invoke("When was Michael Jordan born?")
print("Response:", response['result'])
response = retrieval_qa.invoke("When was Michael Jordan born? What about Lady Gaga?")
print("Response:", response['result'])

# Agentic Run
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

agent = initialize_agent(
	tools,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)

response = agent.run("When was Lady Gaga born?")
print(response)