# Langchain code to Implement Character, Recursive Character, NLTK, Spacy, Markdown and Token Text Splitters.
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, NLTKTextSplitter, SpacyTextSplitter, MarkdownTextSplitter, TokenTextSplitter


loader = PyPDFLoader("Module-4: Keeping Knowledge Organized with indexes/docs/realarmaansidhu.pdf")
pages = loader.load_and_split()


chars = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20).split_documents(pages)
recs = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10, length_function=len).split_documents(pages)

# For NLTK and Spacy, pass the text content, not the list of page objects
all_text = "\n".join([page.page_content for page in pages])
nltks = NLTKTextSplitter(chunk_size=500).split_text(all_text)
spacys = SpacyTextSplitter(chunk_size=500, chunk_overlap=20).split_text(all_text)
mds = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0).create_documents(all_text)
tokens = TokenTextSplitter(chunk_size=100, chunk_overlap=50).split_text(all_text)

def print_splitter_results(name, docs):
	print(f"\n{name} Splitter:")
	if isinstance(docs, list):
		print(f"Total Chunks: {len(docs)}")
		for i, doc in enumerate(docs[:3]):  # Show only first 3 chunks for brevity
			print(f"Chunk {i+1}: {str(doc)[:100]}...")
		if len(docs) > 3:
			print("...")
	else:
		print(docs)

print_splitter_results("Character", chars)
print_splitter_results("Recursive Character", recs)
print_splitter_results("NLTK", nltks)
print_splitter_results("Spacy", spacys)
print_splitter_results("Markdown", mds)
print_splitter_results("Token", tokens)