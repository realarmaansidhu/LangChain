# Langchain code to demonstrate summarization chain using groq LLM from user input text
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

# Initialize the Groq LLM & create a summarization chain
llm = Groq(model_name="llama-3.3-70b-versatile", temperature=0.1)
# Summarization chain using the "stuff" method for less text
# summarization_chain = load_summarize_chain(llm, chain_type="stuff")

# Summarization chain using the "map_reduce" method for larger text
summarization_chain = load_summarize_chain(llm, chain_type="map_reduce")

# Example text to summarize
# text = input("Enter the text you want to summarize: ").strip()
# doc = Document(page_content=text, metadata={"Author":"Armaan Sidhu", "Date Created":"July 07, 2025"})

# Example code to read from a PDF file instead, using pyPDFLoader
loader = PyPDFLoader("Module-2: Large Language Models and LangChain/Docs/Canada.pdf")
docs = loader.load()


# Invoke the summarization chain
response = summarization_chain.invoke({"input_documents": docs})
print("\n--- SUMMARIZING TEXT ---\n")
print("Summary:", response["output_text"].strip())