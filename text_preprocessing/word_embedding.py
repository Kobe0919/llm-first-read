"""
Embedding layer

- Mapping each token ID to the a uniquo vector 
"""
from .sliding_window_with_torch import Build_GPTDataLoader
import torch.nn as nn
import torch

'''
Build a dataloader to generate input-target pairs
'''

txt = "just for test"
max_length = 4
dataloader = Build_GPTDataLoader(
    txt= txt,max_length= max_length, stride= max_length,
    batch_size= 8, shuffle= False,
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# print(inputs)  # DEBUG: temporary output to verify the dataloader
# print(inputs.shape)

'''
Build a token embedding layer
'''
vocab_size = 50257  # GPT-2 vocabulary size
output_dim = 256
token_embedding_layer = nn.Embedding(
    num_embeddings= vocab_size,
    embedding_dim= output_dim,
)
token_embeddings = token_embedding_layer(inputs)

# print(token_embeddings.shape)  # DEBUG: temporary output to verify the embedding layer

'''
Build a position embedding layer
'''

pos_embedding_layer = nn.Embedding(
    num_embeddings= max_length,
    embedding_dim= output_dim,
)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))


'''
input_embedding = token_embedding + position_embedding
'''

input_embeddings = token_embeddings + pos_embeddings

# print(input_embeddings.shape)  # DEBUG: temporary output to verify the final input embeddings