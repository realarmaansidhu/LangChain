# Langchain code to show PyPDF, TextLoader, Selenium URL Loader and Google Drive Sync through Grok LLM
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, SeleniumURLLoader
from langchain_community.document_loaders import GoogleDriveLoader
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9)

text_loader = TextLoader("Module-4: Keeping Knowledge Organized with indexes/docs/realarmaansidhu.txt")
documents = text_loader.load()

pdf_loader = PyPDFLoader("Module-4: Keeping Knowledge Organized with indexes/docs/realarmaansidhu.pdf")
pages = pdf_loader.load_and_split()

selenium_loader = SeleniumURLLoader(urls=["https://realarmaansidhu.com"])
data = selenium_loader.load()

drive_loader = GoogleDriveLoader(
    credentials_path="Module-4: Keeping Knowledge Organized with indexes/credentials.json",
    file_ids=["1rcZ-TaJot0IOt1awhB_gTV7YgDE-AaiH"]
)

docs = drive_loader.load()

print("\n--- TextLoader Output ---")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: Source: {doc.metadata.get('source', 'N/A')}")
    print(f"Preview: {doc.page_content[:200]}...\n")

print("\n--- PyPDFLoader Output ---")
for i, page in enumerate(pages):
    print(f"Page {i+1}: Source: {page.metadata.get('source', 'N/A')}")
    print(f"Preview: {page.page_content[:200]}...\n")

print("\n--- SeleniumURLLoader Output ---")
for i, doc in enumerate(data):
    print(f"URL {i+1}: Source: {doc.metadata.get('source', 'N/A')}")
    print(f"Preview: {doc.page_content[:200]}...\n")

print("\n--- GoogleDriveLoader Output ---")
for i, doc in enumerate(docs):
    print(f"Document {i+1}: Source: {doc.metadata.get('source', 'N/A')}")
    print(f"Preview: {doc.page_content[:200]}...\n")