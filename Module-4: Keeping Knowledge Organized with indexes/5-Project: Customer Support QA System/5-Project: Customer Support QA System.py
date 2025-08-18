# Langchain code to implement a Customer Support QA System using Selenium Loader to parse some URLs and Deeplake to store the vector embeddings, implementing a RAG later on to retrieve answers.
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(dotenv_path="./.env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

urls = ['https://beebom.com/what-is-nft-explained/',
        'https://beebom.com/how-delete-spotify-account/',
        'https://beebom.com/how-download-gif-twitter/',
        'https://beebom.com/how-use-chatgpt-linux-terminal/',
        'https://beebom.com/how-delete-spotify-account/',
        'https://beebom.com/how-save-instagram-story-with-music/',
        'https://beebom.com/how-install-pip-windows/',
        'https://beebom.com/how-check-disk-usage-linux/']

loader = SeleniumURLLoader(urls=urls)
all_docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(all_docs)

my_activeloop_org_id = "armaansidhu"
my_activeloop_dataset_name = "5_Project_Customer_Support_QA_System"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our Deep Lake dataset
db.add_documents(docs)


# implement RetrievalQA using the new API
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    chain_type="stuff"
)
input_query = "What is an NFT?"
result = retrieval_qa.invoke(input_query)
print(result['result'])