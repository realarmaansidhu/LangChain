# Langchain Code to implement an Agent with the Requests_All tool to connect with a Mockup RESTful Server

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv(dotenv_path=".env")
groq_api_key=os.getenv("GROQ_API_KEY")

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9, verbose=True)

tools = load_tools(["requests_all"], llm=llm, allow_dangerous_tools=True)

agent = initialize_agent(llm=llm, tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

url = input("Enter the API URL to work with: ")
while True:
    user_input = input("What do you want to do? (Type 'quit' or 'exit' to stop): ")
    if user_input.lower() in ["quit", "exit"]:
        print("Exiting agent.")
        break
    prompt = f"{user_input} (URL: {url})"
    response = agent.invoke({"input": prompt})
    print(response.get("output", response))