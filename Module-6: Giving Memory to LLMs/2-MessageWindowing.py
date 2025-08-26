# Langchain code to implement ConversationBufferWindowMemory using LCELExpressions with Grok's LLM
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage

load_dotenv(dotenv_path=".env")
print(os.getenv("GROQ_API_KEY"))

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9, verbose=True)

messages = [SystemMessage(content="You are a helpful assistant.")]

WINDOW_SIZE = 5

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    messages.append(HumanMessage(content=user_input))
    prompt = ChatPromptTemplate.from_messages(messages[-WINDOW_SIZE:])
    chain = prompt | llm
    response = chain.invoke({})
    print("Bot:", response.content)
    messages.append(AIMessage(content=response.content if hasattr(response, 'content') else str(response)))