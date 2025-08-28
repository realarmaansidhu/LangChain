# Langchain Code to run a basic agent with Grok's LLM and SerpAPI
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv(dotenv_path=".env")
groq_api_key=os.getenv("GROQ_API_KEY")
serpapi_api_key=os.getenv("SERPAPI_KEY")
print("GROQ_API_KEY:", groq_api_key)
print("SERPAPI_KEY:", serpapi_api_key)

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9, verbose=True)

agent = initialize_agent(llm=llm, tools=load_tools(["serpapi", "requests_all"], serpapi_api_key=serpapi_api_key, llm=llm, allow_dangerous_tools=True), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

response = agent.run(input("What is your question? "))
print(response)