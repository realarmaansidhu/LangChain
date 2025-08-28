# Langchain Code to implement an AI Agent with Google Search and Python-REPL capabilities as tools using Grok's LLM
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_experimental.utilities import PythonREPL
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv(dotenv_path=".env")
groq_api_key=os.getenv("GROQ_API_KEY")
google_cse_id=os.getenv("GOOGLE_CSE_ID")
google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY")

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9, verbose=True)

google_search = GoogleSearchAPIWrapper(
    google_cse_id=google_cse_id,
    google_api_key=google_api_key
)
python_repl = PythonREPL()

toolkit = [
    Tool(
        name="Google-Search",
        func=google_search.run,
        description="Useful for when you need to answer questions about current events or find specific information online. Input should be a search query."
    ),
    Tool(
        name="Python-REPL",
        func=python_repl.run,
        description="Useful for when you need to perform calculations, data analysis, or any Python-related tasks. Input should be a valid Python expression or code snippet."
    )
]

agent = initialize_agent(llm=llm, tools=toolkit, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

while True:
    user_input = input("What is your Demand? (Type 'quit' or 'exit' to stop): ")
    if user_input.lower() in ["quit", "exit"]:
        print("Exiting agent.")
        break
    prompt = f"{user_input}"
    response = agent.invoke({"input": prompt})
    print(response.get("output", response))