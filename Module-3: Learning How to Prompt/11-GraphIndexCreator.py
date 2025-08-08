#Langchain Code for implementing a Graph Index Creator with Groq
import os
from dotenv import load_dotenv
from langchain_community.graphs.index_creator import GraphIndexCreator
from langchain_groq import ChatGroq as Groq

# Load environment variables from .env
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9)

texts = [
    "Paris is the capital of France and the Eiffel Tower is located there.",
    "Tesla was founded by Elon Musk and is a leading electric car manufacturer."
]

graph_index_creator = GraphIndexCreator(llm=llm)

# Use from_text (not from_texts)
combined_text = "\n".join(texts)
knowledge_graph = graph_index_creator.from_text(combined_text)

print("Nodes in the graph:")
print(list(knowledge_graph._graph.nodes))

print("\nEdges in the graph:")
print(list(knowledge_graph._graph.edges(data=True)))