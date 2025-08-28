# Add missing requests import
import requests
# Langchain Code to implement an AI Agent with Wolfram Alpha using Grok's LLM
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq as Groq
from langchain.utilities import WolframAlphaAPIWrapper
from langchain.agents import initialize_agent, AgentType, Tool

load_dotenv(dotenv_path=".env")
groq_api_key = os.getenv("GROQ_API_KEY")
wolfram_app_id = os.getenv("WOLFRAM_ALPHA_APPID")
print("Grok API Key:", groq_api_key)
print("Wolfram App ID:", wolfram_app_id)

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9, verbose=True)

def wolfram_short_answer(query: str) -> str:
    url = "http://api.wolframalpha.com/v1/result"
    params = {"i": query, "appid": wolfram_app_id}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code == 200:
        return r.text
    else:
        return f"Error: {r.status_code} {r.text}"

wolfram_tool = Tool(
    name="WolframAlpha",
    func=wolfram_short_answer,
    description="Get factual answers from Wolfram Alpha"
)

agent = initialize_agent(llm=llm, tools=[wolfram_tool], agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, description="An AI agent that uses Wolfram Alpha to answer questions.")

while True:
    user_input = input("Enter your question (or type 'quit' or 'exit' to stop): ")
    if user_input.lower() in ["quit", "exit"]:
        print("Exiting agent.")
        break
    response = agent.invoke({"input": user_input})
    print(response.get("output", response))