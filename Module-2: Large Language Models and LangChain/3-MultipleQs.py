# Langchain code to answer multiple questions using a single LLM call
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv(dotenv_path=".env")
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print("Loaded HuggingFace Access Token: ", HUGGINGFACEHUB_ACCESS_TOKEN)

request = input("Enter your questions separated by commas: ")

prompt = PromptTemplate(
    template="Answer each of the following questions separately and concisely (one line answers):\n"
        "{questions}",
    input_variables=["questions"]
)

llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta", # This one **is** hosted
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN,
    temperature=0.1
)
chain = prompt | llm
response = chain.invoke({"questions": request})
print("Response: ", response)