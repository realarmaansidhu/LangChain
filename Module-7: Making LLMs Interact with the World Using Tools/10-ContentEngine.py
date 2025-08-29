# Langchain Code to Implement an AI Agent as a Content Engine that uses LCEL Chain as a tool to generate Content
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path=".env")
groq_api_key=os.getenv("GROQ_API_KEY")
print("Grok API Key:", groq_api_key)

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9, verbose=True)

prompt = ChatPromptTemplate.from_template("Write a detailed article about {topic}.")

lcel_chain = prompt | llm | StrOutputParser()  # this is a real LCEL chain

lcel_tool = Tool(
    name="LCEL Chain",
    func=lambda topic: lcel_chain.invoke({"topic": topic}),  # still callable
    description="Generates content using a real LCEL chain"
)

agent = initialize_agent(
    llm=llm,
    tools=[lcel_tool],
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
