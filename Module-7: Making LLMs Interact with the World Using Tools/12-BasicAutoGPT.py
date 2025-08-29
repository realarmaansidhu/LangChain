# Langchain code to implement AutoGPT with Google's Embeddings and Locally Hosted Ollama Server
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM as Ollama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.agents import Tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.file_management import WriteFileTool, ReadFileTool
from langchain_google_community import GoogleSearchAPIWrapper

load_dotenv(dotenv_path=".env")
google_cse_id=os.getenv("GOOGLE_CSE_ID")
google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = DeepLake(dataset_path="Databases/Module-7/12_BasicAutoGPT", embedding_function=embeddings)
print("****Environment Variables Loaded Successfully****")

# set the Task
while True:
    task = input("Enter Your Task: ").strip()
    if not task:
        print("Warning: Task cannot be empty. Please enter a valid task.")
        continue
    def google_search_wrapper(tool_input: str):
        return GoogleSearchAPIWrapper(
            google_api_key=google_api_key,
            google_cse_id=google_cse_id
        ).run(tool_input)

    google_search_tool = Tool(
        name="google_search",
        description="Search Google for recent results.",
        func=google_search_wrapper
    )

    toolkit = [google_search_tool, WriteFileTool(), ReadFileTool()]

    # create the AutoGPT agent
    auto_gpt = AutoGPT.from_llm_and_tools(
        llm=Ollama(model="mistral", base_url="http://localhost:11434"),
        ai_name="Agent-X",
        ai_role="AI Assistant",
        tools=toolkit,
        memory=vector_db.as_retriever(),
    )
    auto_gpt.chain.verbose = True

    response = auto_gpt.run([task])
    print(response)
    
    if task.lower() in ["quit", "exit"]:
        print("Exiting agent.")
        break