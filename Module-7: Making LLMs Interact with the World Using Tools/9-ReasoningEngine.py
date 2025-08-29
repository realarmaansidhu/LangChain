# Langchain Code to implement an AI Agent as a Reasoning Engine with Wolfram Alpha, Google Search and Python-REPL capabilities as tools using Grok's LLM
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv(dotenv_path=".env")
groq_api_key=os.getenv("GROQ_API_KEY")
google_cse_id=os.getenv("GOOGLE_CSE_ID")
google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY")
wolfram_app_id=os.getenv("WOLFRAM_ALPHA_APPID")

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9, verbose=True)

toolkit = load_tools(
    ["google-search", "llm-math"],
    google_cse_id=google_cse_id,
    google_api_key=google_api_key,
    llm=llm
)

custom_prompt = (
    "You are a reasoning agent with access to tools. "
    "When you have the answer, always respond with 'Final Answer: <your answer>'. "
    "Use tools as needed."
)

agent = initialize_agent(
    llm=llm,
    tools=toolkit,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

while True:
    if user_input := input("What is your Demand? (Type 'quit' or 'exit' to stop): "):
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting agent.")
            break
        prompt = f"{user_input}"
        response = agent.invoke({"input": prompt})
        print(response.get("output", response))