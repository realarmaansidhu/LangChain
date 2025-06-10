import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv(dotenv_path="/.env")  # This loads variables from .env into the environment

print("Loaded API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.9
)

product = "Tomatos"

prompt = PromptTemplate(
    input_variables=["product"],
    template="Suggest a unique, catchy, one-word name for a company that makes {product}. Respond with only the name, nothing else."
)

chain = prompt | llm

response = chain.invoke({"product": product})
print("Generated Company Name: ", response.content)