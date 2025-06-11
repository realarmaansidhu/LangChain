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

# create our documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "armaansidhu" 
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our Deep Lake dataset
db.add_documents(docs)

retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm,
	chain_type="stuff",
	retriever=db.as_retriever()
)
# run the retrieval QA chain
response = retrieval_qa.invoke("When was Napoleon Bonaparte born?")
print("Response:", response['result'])
response = retrieval_qa.invoke("When was Louis XIV born?")
print("Response:", response['result'])
response = retrieval_qa.invoke("When was Louis XIV born? What about Napoleon Bonaparte?")
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

response = agent.run("When was Napoleone born?")
print(response)