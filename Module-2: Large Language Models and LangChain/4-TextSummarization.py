# Langchain code to summarize text through a chain using a single LLM call
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain_core.prompts import PromptTemplate

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

text = input("Enter the text to summarize: ")
prompt = PromptTemplate(
    template="Summarize the following text in a concise manner:\n"
        "{text}",
    input_variables=["text"]
)

llm = Groq(model_name="llama-3.3-70b-versatile", temperature=0.9)
prompt_chain = prompt | llm
response = prompt_chain.invoke({"text": text})
print("Summary:", response.content.strip())