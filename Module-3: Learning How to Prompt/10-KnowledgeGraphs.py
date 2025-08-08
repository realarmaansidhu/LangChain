import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq as Groq
import networkx as nx
from pyvis.network import Network
import re

# Load environment variables from .env file
load_dotenv()

# === Step 0: Get Groq API key from environment variable ===
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# === Step 1: Define prompt template for triple extraction ===
DELIMITER = "<|>"

prompt_template = """
Extract only clear, factual knowledge triples (subject, predicate, object) from the text below.
Each triple should be in the format: (subject, predicate, object).
- The subject and object should be named entities or concrete nouns (not phrases or pronouns).
- The predicate should be a meaningful relation phrase (e.g., "is the capital of", "is located in"), not just "is" or "is located".
- Do NOT use phrases like "the capital of France" or pronouns like "there" as subject or object.
Output ALL triples in a SINGLE LINE, separated ONLY by <|> (no newlines, no extra text).
If no triples exist, respond with NONE.

Text:
{text}

Output:
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=prompt_template
)

# === Step 2: Initialize Groq model and chain ===
llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9)
chain = prompt | llm

# === Step 3: Define helper functions ===
def parse_triples(text):
    triple_pattern = re.compile(r"\((.*?),\s*(.*?),\s*(.*?)\)")
    return ["(" + ", ".join(match) + ")" for match in triple_pattern.findall(text)]

def is_named_entity(entity):
    # Ignore pronouns and long phrases
    pronouns = {"there", "it", "here"}
    if entity.lower() in pronouns:
        return False
    if len(entity.split()) > 3:
        return False
    if " of " in entity or " in " in entity:
        return False
    return True

def is_good_predicate(predicate):
    # Ignore too generic predicates
    generic_predicates = {"is", "are", "was", "were", "be", "located", "located in"}
    if predicate.strip().lower() in generic_predicates:
        return False
    if len(predicate.strip().split()) < 3:  # e.g., "is the capital of"
        return False
    return True

def create_graph(triples):
    G = nx.DiGraph()
    triple_pattern = re.compile(r"\((.*?),\s*(.*?),\s*(.*?)\)")
    for triple in triples:
        match = triple_pattern.match(triple)
        if match:
            subject, predicate, obj = match.groups()
            subject, obj, predicate = subject.strip(), obj.strip(), predicate.strip()
            if is_named_entity(subject) and is_named_entity(obj) and is_good_predicate(predicate):
                G.add_edge(subject, obj, label=predicate)
        else:
            print(f"Skipping malformed triple: {triple}")
    return G

def visualize_graph(G, output_file="knowledge_graph.html"):
    net = Network(notebook=True, height="600px", width="100%")
    net.from_nx(G)
    net.toggle_physics(True)
    net.show_buttons(filter_=['edges'])
    net.show(output_file)

# === Step 4: Input text for triple extraction ===
input_text = "Paris is the capital of France and the Eiffel Tower is located there."

# === Step 5: Run chain to extract triples ===
response = chain.invoke({"text": input_text})
print("Raw output:")
print(response.content)

# === Step 6: Parse triples ===
triples = parse_triples(response.content)
print("\nParsed triples:")
for t in triples:
    print(t)

# === Step 7: Build graph and visualize ===
graph = create_graph(triples)
visualize_graph(graph)