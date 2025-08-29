# Langchain Code to implement BabyAGI with Google's Embeddings
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain_experimental.autonomous_agents import BabyAGI

load_dotenv(dotenv_path=".env")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = DeepLake(dataset_path="Databases/Module-7/11_BasicBabyAGI", embedding_function=embeddings)

# set the goal
goal = "Plan a trip to the Grand Canyon"

# create thebabyagi agent
# If max_iterations is None, the agent may go on forever if stuck in loops
baby_agi = BabyAGI.from_llm(
    llm= Ollama(model="mistral", base_url="http://localhost:11434"),
    vectorstore=vector_db,
    verbose=False,
    max_iterations=3
)
response = baby_agi({"objective": goal})