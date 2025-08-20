# LangChain code to Implement LCELChain for LLMChain with batch processing, Parsers, Conversational Chain, Sequential Chain

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.memory import ConversationBufferMemory

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9)

# LLM Chain
llmchain_prompt = PromptTemplate(
    template="Replace the following user supplied word {input_word} with another closely matching word based on the following user supplied word context {word_context}. NOTE: JUST PRINT ONE WORD, THE REPLACEMENT NOTHING ELSE NOT EVEN A PUNCTUATION MARK",
    input_variables=["input_word", "word_context"]
)
llmchain_output = llmchain_prompt | llm
print(llmchain_output.invoke({"input_word": "fan", "word_context": "an object that creates a current of air"}).content)

# With Batch Processing & .predict()
input_list = [{"input_word": "heater", "word_context":
               "an object that produces heat"},
               {"input_word": "cooler", "word_context": "an object that cools air"},
               {"input_word": "TV", "word_context": "an object that displays video content"}]

results = llmchain_output.batch(input_list)
for res in results:
    print(res.content)

# With Parser
output_parser = CommaSeparatedListOutputParser()
parsed_results = [output_parser.parse(res.content) for res in results]
print(parsed_results)

# Conversational Chain
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Basic conversational prompt
conversational_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="The following is a conversation between a user and an AI assistant.\n\n{history}\nUser: {input}\nAI:"
)

conversation_chain = conversational_prompt | llm

def chat(input_text):
    history = memory.load_memory_variables({})["history"]
    response = conversation_chain.invoke({"history": history, "input": input_text}).content
    memory.save_context({"input": input_text}, {"output": response})
    return response

print("User: Hello!")
print("AI:", chat("Hello!"))

print("User: Tell me a joke.")
print("AI:", chat("Tell me a joke."))

# Sequential Chain
prompt_chain_1 = PromptTemplate(
    input_variables=["animal"],
    template="Give a fun, interesting fact about the following animal: {animal}"
)
chain_1 = prompt_chain_1 | llm

prompt_chain_2 = PromptTemplate(
    input_variables=["fact"],
    template="Based on this fact: '{fact}', write a simple question a child might ask to learn more about it."
)
chain_2 = prompt_chain_2 | llm

overall_chain = chain_1 | chain_2

animal_name = "giraffe"
question = overall_chain.invoke({"animal": animal_name})

print("Generated question:", question.content)