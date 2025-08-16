#Langchain Grok Code to use a Textloader to load files, split them into chunks and upload to deeplake to implement a RetrievalQA

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DeepLake
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq as Groq
from dotenv import load_dotenv
from langchain.chains import RetrievalQA

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9)

text="""Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google is offering developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses “generate text, images, code, videos, audio, and more from
simple natural language prompts.”"""

with open("my_file.txt", "w") as f:
    f.write(text)

loader = TextLoader("my_file.txt")
docs_from_file = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(docs_from_file)
print(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db = DeepLake(dataset_path="hub://armaansidhu/Module-4", embedding_function=embeddings)
db.add_documents(docs)

retriever = db.as_retriever()

# # Simple Contextual Retrieving:
# query = "What is PaLM and what is Google offering?"
# results = retriever.get_relevant_documents(query)
# print("Relevant documents:", results)

# RAG Retrieving:
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
query = "How is Google challenging OpenAI?"
response = qa_chain.run(query)
print("Answer:", response)