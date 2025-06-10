import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

load_dotenv(dotenv_path="/.env")  # This loads variables from .env into the environment

print("Loaded AI API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.9
)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# Start the conversation
conversation.predict(input="Tell me about yourself.")

# Continue the conversation
conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with data analysis?")

# Display the conversation
print(conversation)