# Langchain Code to Implement HumanMessagePromptTemplate, SystemMessagePromptTemplate, and AIMessagePromptTemplate using Mistral LLM
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI as Mistral
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(dotenv_path=".env")
print("Loaded Mistral API key:", os.getenv("MISTRAL_API_KEY"))

llm = Mistral(model_name="mistral-tiny", temperature=0.7)
hmn_msg = input("Enter your question: ").strip()

messages = [
    SystemMessage(content="You are a helpful assistant that provides concise answers to user questions."),
    HumanMessage(content=hmn_msg)
]

prompt = HumanMessage(
    content="I'd like to know more about the city you just mentioned."
)
# add to messages
messages.append(prompt)

# Invoke the LLM with the messages
ai_message = llm.invoke(messages).content.strip()
print(ai_message)
print("-" * 50)