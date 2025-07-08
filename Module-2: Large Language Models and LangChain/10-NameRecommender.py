# Langchain code to recommend a simple Name for a Cmmpany based on what they do using Gemini 2.5 Flash
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv(dotenv_path=".env")
print("Loaded Gemini API key:", os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.9)

# Get user input for company description
company_description = input("Describe your company: ").strip()

prompt = PromptTemplate(
    input_variables=["company_description"],
    template= "Suggest a creative company name for a business that does the following (ONLY ONE NAME TO BE SUGGESTED!): {company_description}"
)

# Create the chain
chain = prompt | llm
print("Recommended Name:", chain.invoke({"company_description": company_description}).content.strip())