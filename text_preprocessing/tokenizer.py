'''
Tokenizer

1. Segmentation
2. Mapping to token ID

'''


import tiktoken  # BPE algorithm by GPT models


text = "just for test"

# Build a tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Encode the text
integers = tokenizer.encode(text)
print(integers)

# Decode the integers
strings = tokenizer.decode(integers)
print(strings)