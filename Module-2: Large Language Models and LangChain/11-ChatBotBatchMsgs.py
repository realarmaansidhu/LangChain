# Langchain code to demonstrate Batch Messages to a Chat Model
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as groq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(dotenv_path=".env")
print("Loaded Groq API key:", os.getenv("GROQ_API_KEY"))

llm = groq(model_name="llama-3.3-70b-versatile", temperature=0.9)

'''
# Define the system message
system_message = "You are a helpful assistant that provides concise answers to user questions."

# Define the human message 
human_message = input("Enter your question: ").strip()

# Create the system and human messages
messages = [
    SystemMessage(content=system_message),
    HumanMessage(content=human_message)
]
'''

# Passing batch messages to the LLM using a while loop to take user input
human_messages = []
while True:
    human_message = input("Enter your question (or type 'exit' to quit): ").strip()
    if human_message.lower() == 'exit':
        break
    human_messages.append(HumanMessage(content=human_message))

system_message = "You are a helpful assistant that provides concise answers to user questions."

for human_message in human_messages:
    messages = [
        SystemMessage(content=system_message),
        human_message
    ]
    # Define the AI message
    ai_message = llm.invoke(messages).content.strip()
    print("AI Response:", ai_message)
    print("-" * 50)