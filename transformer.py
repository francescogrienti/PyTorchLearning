import torch
import torch.nn as nn


# TODO understand meaning of torch.einsum (tensor multiplication with Einstein notation probably?)
# TODO dive deeper in the multi-head variant of the Attention Mechanism.
# TODO finish implementing the entire Transformer (encoder and decoder blocks).

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * self.heads == self.embed_size)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # Batch size (for example, in a text the number of sentences)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[
            1]  # Number of tokens in each sentence (basically, sequence length).
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
