# Langchain Code to run a basic agent with Grok's LLM and Google Search API
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv(dotenv_path=".env")
groq_api_key=os.getenv("GROQ_API_KEY")
google_cse_id=os.getenv("GOOGLE_CSE_ID")
google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY")

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9, verbose=True)

tools = load_tools(["google-search"], google_cse_id=google_cse_id,
google_api_key=google_api_key,
llm=llm)

agent = initialize_agent(llm=llm, tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

response = agent.invoke({"input": input("What is your question? ")})
print(response)