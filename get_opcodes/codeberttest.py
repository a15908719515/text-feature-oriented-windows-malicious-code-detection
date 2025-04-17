from transformers import AutoTokenizer, AutoModel
import torch


# Init
tokenizer = AutoTokenizer.from_pretrained("E:\pycharmcode\codebert-base")
model = AutoModel.from_pretrained("E:\pycharmcode\codebert-base")


# Tokenization
nl_tokens = tokenizer.tokenize("return maximum value")
code_tokens = tokenizer.tokenize("def max(a,b): if a>b: return a else return b")
tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]


# Convert tokens to ids
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
context_embeddings = model(torch.tensor(tokens_ids)[None,:])[0]


# Print
print(context_embeddings)