# Langchain code to demonstrate a simple role prompting example with Groq LLM
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9)

# Define a role template
template = """
You are a rude and sarcastic assistant.
You will answer the user's question in a sarcastic manner. Use short sentences and be very direct, you can use :), ;), !! etc. to make your point.\n
"""

# Create a prompt with the role template
user_input = input("Enter your question: ")
prompt = template + user_input
response = llm.invoke(prompt)
print("Response:", response.content.strip())
print("-" * 50)