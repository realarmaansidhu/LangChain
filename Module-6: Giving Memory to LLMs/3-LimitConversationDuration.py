# Langchain code to implement ConversationBufferWindowMemory with a Limit to Turns with Grok's LLM
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage

load_dotenv(dotenv_path=".env")
print(os.getenv("GROQ_API_KEY"))

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9, verbose=True)

messages = [SystemMessage(content="You are a helpful assistant.")]

MAX_TURNS = 20
turn_count = 0

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    messages.append(HumanMessage(content=user_input))
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm
    response = chain.invoke({})
    print("Bot:", response.content)
    messages.append(AIMessage(content=response.content if hasattr(response, 'content') else str(response)))
    turn_count += 1
    if turn_count >= MAX_TURNS:
        print("Max turns reached. Ending conversation.")
        break