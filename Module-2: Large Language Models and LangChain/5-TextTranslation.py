# Langchain code to translate text through a chain using a single LLM call
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain_core.prompts import PromptTemplate

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

text = input("Enter the text to translate: ")
target_language = input("Enter the target language (e.g., French, Spanish): ")

prompt = PromptTemplate(
    template="Translate the following text to {target_language}:\n"
        "{text}",
    input_variables=["text", "target_language"]
)

llm = Groq(model_name="llama-3.3-70b-versatile", temperature=0.1)
prompt_chain = prompt | llm
response = prompt_chain.invoke({"text": text, "target_language": target_language})
print("Translation:", response.content.strip())