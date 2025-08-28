# Langchain Code to pass a Python-REPL to an Agent that uses Grok's LLM to compute complex Algorithmic tasks using Python
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_experimental.utilities import PythonREPL

load_dotenv(dotenv_path=".env")
groq_api_key = os.getenv("GROQ_API_KEY")
print("GROQ_API_KEY:", groq_api_key)

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9, verbose=True)

python_repl = PythonREPL()

tools = [Tool(name="python_repl", description="Useful for when you need to run python code to compute a result, or when you need to execute code to transform data.", func=python_repl.run)]

agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)

while True:
    user_input = input("What do you want to do? (Type 'quit' or 'exit' to stop): ")
    if user_input.lower() in ["quit", "exit"]:
        print("Exiting agent.")
        break
    prompt = f"{user_input}"
    response = agent.invoke({"input": prompt})
    print(response.get("output", response))