import os
import tempfile
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI as Mistral
from pyvis.network import Network
import networkx as nx
import re
import streamlit as st
import streamlit.components.v1 as components

# Load environment variables
load_dotenv()
print("Loaded Mistral API key:", os.getenv("MISTRAL_API_KEY"))

# LLM and prompt
llm = Mistral(model_name="mistral-tiny", temperature=0.7)
prompt_template = """
Extract all possible factual knowledge triples from the text below.
Each triple must be in the format: (subject, predicate, object).
- Include ALL types of factual relationships, not just the most important ones.
- Include facts about education, early life, family, organizations, awards, locations, and any other concrete relationships.
- The subject and object should be real, concrete named entities (people, places, organizations, etc.), not pronouns, generic phrases, or placeholders like "NONE", "N/A", "unknown", or empty strings.
- The predicate should be a meaningful relationship (e.g., "is the capital of", "founded", "located in", "graduated from", "was born in", "is a member of").
- Do NOT include any triple where the subject or object is missing, "NONE", "N/A", "unknown", or not a real entity.
- Do NOT use vague predicates like "is", "was", or "has".
- Do NOT use pronouns ("he", "she", "it", "they") or generic terms ("there", "someone") as subject or object.
- If no valid triples exist, respond with NONE.
- Output ALL triples in a SINGLE LINE, separated ONLY by <|> (no newlines, no extra text).

Text:
{text}

Output:
"""
prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
chain = prompt | llm

normalization_prompt = PromptTemplate(
    input_variables=["original_text", "triples"],
    template="""
Input Text:
{original_text}

Extracted triples:
{triples}

Task:
Unify all entities in the triples that refer to the same real-world entity, using the context of the input text.
Output the updated triples in the same format: (subject, predicate, object), separated by commas.
If no changes needed, output the triples as-is.
"""
)
normalization_chain = normalization_prompt | llm

# Streamlit UI
st.title("Knowledge Tree Web App")
st.write("Paste your text below to extract factual triples and visualize the knowledge graph.")
user_text = st.text_area("Paste your text:", height=200)

def parse_triples(text):
    triple_pattern = re.compile(r"\((.*?),\s*(.*?),\s*(.*?)\)")
    return [match for match in triple_pattern.findall(text)]

def chunk_text(text, max_chunk_size=1000):
    # Split text into chunks of max_chunk_size characters, trying to split at sentence boundaries
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks
# *
def split_text_for_summarization(text, n_chunks=10):
    sentences = re.split(r'(?<=[.!?]) +', text)
    avg = len(sentences) // n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * avg
        end = (i + 1) * avg if i < n_chunks - 1 else len(sentences)
        chunk = " ".join(sentences[start:end])
        if chunk.strip():
            chunks.append(chunk)
    return chunks
# *
SUMMARY_CHUNK_THRESHOLD = 10

if st.button("Generate Knowledge Graph"):
    if not user_text.strip():
        st.warning("Please enter some text to process.")
    else:
        chunks = chunk_text(user_text)
        if len(chunks) > SUMMARY_CHUNK_THRESHOLD:
            st.write("Summarizing the Input ...")
            summary_chunks = []
            split_chunks = split_text_for_summarization(user_text, n_chunks=SUMMARY_CHUNK_THRESHOLD)
            summary_prompt = PromptTemplate(
                input_variables=["text"],
                template="Summarize the following text in detail, preserving all key facts, events, people, organizations, and relationships:\n\n{text}\n\nSummary:"
            )
            summary_chain = summary_prompt | llm
            for i, chunk in enumerate(split_chunks):
                summary_response = summary_chain.invoke({"text": chunk})
                summary_text = summary_response.content
                summary_chunks.append(summary_text)
            st.write("Summarized the Input ✅")
            chunks = summary_chunks
        all_triples_set = set()
        with st.spinner("Processing text and extracting triples..."):
            for i, chunk in enumerate(chunks):
                st.write(f"Processing chunk {i+1} of {len(chunks)} ✅")
                response = chain.invoke({"text": chunk})
                extracted_triples = response.content
                print("**Raw LLM output for chunk:**", extracted_triples)
                triples = parse_triples(extracted_triples)
                for triple in triples:
                    all_triples_set.add(triple)
        if not all_triples_set:
            st.info("No valid triples extracted.")
        else:
            triples = list(all_triples_set)
            triples_str = ", ".join([f"({subj}, {pred}, {obj})" for subj, pred, obj in triples])
            normalization_response = normalization_chain.invoke({"original_text": user_text, "triples": triples_str})
            normalized_triples_str = normalization_response.content
            st.write("**Normalized triples output✅**")
            print("**Normalized triples output:**", normalized_triples_str)
            normalized_triples = parse_triples(normalized_triples_str)
            G = nx.DiGraph()
            for subj, pred, obj in normalized_triples:
                G.add_edge(subj.strip(), obj.strip(), label=pred.strip())
            net = Network(height="600px", width="100%", notebook=False, directed=True, cdn_resources='remote')
            net.from_nx(G)
            net.set_options("""
            {
              "physics": {
                "enabled": true,
                "barnesHut": {
                  "springLength": 200,
                  "springConstant": 0.02,
                  "avoidOverlap": 0.5
                }
              },
              "nodes": {
                "font": {
                  "size": 20
                }
              }
            }
            """)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                net.save_graph(tmp_file.name)
                tmp_file_path = tmp_file.name
            with open(tmp_file_path, "r", encoding="utf-8") as f:
                net_html = f.read()
            components.html(net_html, height=800, width=900, scrolling=True)
            with st.expander("Show extracted nodes, edges, and triples"):
                st.write("**Nodes:**", list(G.nodes()))
                st.write("**Edges:**", list(G.edges(data=True)))
                st.write("**Triples:**", normalized_triples)