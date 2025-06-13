# Code to import tokenizers and demonstrate their usage
from transformers import AutoTokenizer
# from transformers import AutoModel 
# import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModel.from_pretrained("gpt2")

text = "Hello, how are you doing today?"
tokens = tokenizer.encode(text, add_special_tokens=True)
decoded_text = tokenizer.decode(tokens)

print("Original Text:", text)
print("Tokens:", tokens)
print("Decoded Text:", decoded_text)

# with torch.no_grad():
#     embeddings = model.transformer.wte(tokens)
# print(embeddings.shape)