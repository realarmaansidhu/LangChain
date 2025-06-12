# Simple Question Answering with the Hugging Face model google/flan-t5-largeCopy
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv(dotenv_path=".env")
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print("Loaded HuggingFace Access Token: ", HUGGINGFACEHUB_ACCESS_TOKEN)

request = input("Enter your question: ")
prompt = PromptTemplate(
    template="Answer the question: {question}",
    input_variables=["question"]
)

llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta", # This one **is** hosted
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN,
    temperature=0.1
)

chain = prompt | llm
response = chain.invoke({"question": request})
print("Response: ", response)