import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq

load_dotenv(dotenv_path="/.env")  # This loads variables from .env into the environment

print("Loaded AI API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.9
)

text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."

response = llm.invoke(text)
print(response.content)