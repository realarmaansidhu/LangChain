# Langchain Code to Implement an AI Agent that uses Wikipedia Tool with Grok's LLM
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv(dotenv_path=".env")
groq_api_key=os.getenv("GROQ_API_KEY")
print("GROQ_API_KEY: ", groq_api_key)

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9, verbose=True)

tools = load_tools(["wikipedia"], llm=llm)

agent=initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

while True:
    user_input = input("What do you want to research about via Wikipedia? (Type 'quit' or 'exit' to stop): ")
    if user_input.lower() in ["quit", "exit"]:
        print("Exiting agent.")
        break
    response = agent.invoke({"input": user_input})
    print(response.get("output", response))