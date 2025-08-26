# Langchain code to implement a ConversationChain that Summarizes Memory using Grok LLM
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage

load_dotenv(dotenv_path=".env")
print(os.getenv("GROQ_API_KEY"))

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9, verbose=True)

# Start with system message
messages = [SystemMessage(content="You are a helpful assistant.")]

def summarize_conversation(messages):
    # Summarize the conversation history
    prompt = ChatPromptTemplate.from_messages([SystemMessage(content="Summarize the following conversation for future context:"), *messages])
    chain = prompt | llm
    response = chain.invoke({})
    return response.content if hasattr(response, 'content') else str(response)

SUMMARY_INTERVAL = 10
summary = None
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

    # Summarize conversation every SUMMARY_INTERVAL turns
    if turn_count % SUMMARY_INTERVAL == 0:
        summary = summarize_conversation(messages)
        print("[Summary updated]:", summary)
        # Optionally, reset messages and use summary as context
        messages = [SystemMessage(content="Conversation summary: " + summary)]