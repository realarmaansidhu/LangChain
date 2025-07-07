# Langchain code to demonstrate basic Q&A using Gemini LLM from user input text
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load environment variables (make sure your Gemini API key is in .env)
load_dotenv(dotenv_path=".env")
print("Loaded Gemini API key:", os.getenv("GOOGLE_API_KEY"))

# Create the prompt template
prompt = PromptTemplate(
    template="Answer the following question concisely:\n{question}",
    input_variables=["question"]
)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.1)

# Create the chain
chain = prompt | llm

# Get user question
question = input("Enter your question: ").strip()

# Run the chain
response = chain.invoke({"question": question})
print(response.content.strip())
