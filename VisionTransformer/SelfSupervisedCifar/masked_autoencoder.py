import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim

"""
In this .py we implement the MAE (MaskedAutoEncoder) with the aim of reconstructing masked images 
from the CIFAR Database. The general architecture of a MAE consists of: masked input, encoder, decoder. We will 
implement the encoder and the decoder exploiting the Attention Mechanism. This work belongs to the Supervised Learning 
general framework. 
"""

"""
CONSTANTS 
"""
EMBED_SIZE = 48
NUM_HEADS = 4
NUM_HIDDEN_LAYERS = 4
FORWARD_EXPANSION = 4 * 48
PATCH_SIZE = 4
NUM_CLASSES = 10
NUM_CHANNELS = 3
QKV_BIAS = True
DROPOUT = 0.0
EPOCHS = 100

"""
CLASSES
"""


class PatchCreation(nn.Module):
    """
    Patches creation module.
    """

    def __init__(self, patch_size, embed_size, num_channels, image_size):
        super().__init__()
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_channels = num_channels
        self.image_size = image_size
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(in_channels=self.num_channels, out_channels=self.embed_size,
                                    kernel_size=self.patch_size,
                                    stride=self.patch_size)

    def forward(self, x):
        x = self.projection(x)
        # The goal of the following operation is to transform the input data into something suitable for a transformer
        # in terms of size ----> (batch_size, num_patches, embed_size).
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchEmbedding(nn.Module):
    """
    Patches embedding module: combination of patches, position and class embeddings.
    """

    def __init__(self, patch_size, embed_size, num_channels, image_size, dropout):
        super().__init__()
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_channels = num_channels
        self.image_size = image_size

        self.embedding = PatchCreation(patch_size, embed_size, num_channels, image_size)
        # Create a learnable [CLS] token
        # Similar to BERT, the [CLS] token is added to the beginning of the input sequence
        # and is used to classify the entire sequence
        # nn.Parameter makes the tensor a learnable parameter of the model (it is updated during the training process)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        # Create position embeddings for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.embedding.num_patches + 1, embed_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.
    """

    def __init__(self, embed_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.embed_size = embed_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(embed_size, attention_head_size, bias=bias)
        self.key = nn.Linear(embed_size, attention_head_size, bias=bias)
        self.value = nn.Linear(embed_size, attention_head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, embed_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        energy = torch.einsum("nqd,nkd->nqk", [query, key])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)
        out = torch.einsum("nql,nld->nqd", [attention, value])
        return out


# TODO Fix the input x, enc_output
class CrossAttentionHead(nn.Module):
    """
    Single-head attention module for cross-attention layer.

    """

    def __init__(self, embed_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.embed_size = embed_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(embed_size, attention_head_size, bias=bias)
        self.key = nn.Linear(embed_size, attention_head_size, bias=bias)
        self.value = nn.Linear(embed_size, attention_head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        energy = torch.einsum("nqd,nkd->nqk", [query, key])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)
        out = torch.einsum("nql,nld->nqd", [attention, value])
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the Encoder module.
    """

    def __init__(self, embed_size, num_attention_heads, dropout, qvk_bias):
        super().__init__()
        self.embed_size = embed_size
        self.num_attention_heads = num_attention_heads
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.embed_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        assert (self.all_head_size == self.embed_size)
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = qvk_bias
        # Create a list of attention heads
        self.dropout = dropout
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.embed_size,
                self.attention_head_size,
                self.dropout,
                self.qkv_bias
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the embed size
        # In most cases, all_head_size and hidden_size are the same
        # Not clear this layer...all_head_size == embed_size, but in the forward method the tensors are concatenated
        # and then projected through a layer of dim = embed_size
        self.output_projection = nn.Linear(self.all_head_size, self.embed_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head: why dim = -1?
        attention_output = torch.cat([attention_output for attention_output in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the embedding size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output
        return attention_output


class CrossMultiHeadAttention(nn.Module):
    """
      Multi-head attention module for cross-attention layer.
    """

    def __init__(self, embed_size, num_attention_heads, dropout, qvk_bias):
        super().__init__()
        self.embed_size = embed_size
        self.num_attention_heads = num_attention_heads
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.embed_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        assert (self.all_head_size == self.embed_size)
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = qvk_bias
        # Create a list of attention heads
        self.dropout = dropout
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = CrossAttentionHead(
                self.embed_size,
                self.attention_head_size,
                self.dropout,
                self.qkv_bias
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the embed size
        # In most cases, all_head_size and hidden_size are the same
        # Not clear this layer...all_head_size == embed_size, but in the forward method the tensors are concatenated
        # and then projected through a layer of dim = embed_size
        self.output_projection = nn.Linear(self.all_head_size, self.embed_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x, enc_output) for head in self.heads]
        # Concatenate the attention outputs from each attention head: why dim = -1?
        attention_output = torch.cat([attention_output for attention_output in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the embedding size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output
        return attention_output


class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, embed_size, forward_expansion, dropout):
        super().__init__()
        self.dense_1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(forward_expansion * embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size, num_attention_heads, dropout, qvk_bias)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, forward_expansion, dropout)

    def forward(self, x):
        # Self-attention
        attention_output = self.attention(self.layer_norm_1(x))
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layer_norm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output
        return x


class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias, num_hidden_layers):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(num_hidden_layers):
            block = EncoderBlock(embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias)
            self.blocks.append(block)

    def forward(self, x, ):
        # Calculate the transformer block's output for each block
        for block in self.blocks:
            x = block(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size, num_attention_heads, dropout, qvk_bias)
        self.cross_attention = CrossMultiHeadAttention(embed_size, num_attention_heads, dropout, qvk_bias)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, forward_expansion, dropout)

    def forward(self, x, enc_output):
        attention_output = self.attention(self.layer_norm_1(x))
        x = x + attention_output
        cross_attention_output = self.cross_attention(x, enc_output)
        x = x + cross_attention_output
        mlp_output = self.mlp(self.layer_norm_2(x))
        x = x + mlp_output
        return x


class Decoder(nn.Module):
    def __init__(self, embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias, num_hidden_layers):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(num_hidden_layers):
            block = DecoderBlock(embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias)
            self.blocks.append(block)

    def forward(self, x, enc_output):
        # Calculate the transformer block's output for each block
        for block in self.blocks:
            x = block(x, enc_output)
        return x


# TODO Fix this class
class MaskedAutoEncoder(nn.Module):
    def __init__(self, embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias, dataset, patch_size,
                 num_hidden_layers, num_channels):
        super().__init__()
        self.image_size = dataset[0][0].shape[-1]
        self.embed_size = embed_size
        # Create the embedding layer
        self.embedding = PatchEmbedding(patch_size, embed_size, num_channels, self.image_size, dropout)
        # Create the encoder module
        self.encoder = Encoder(embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias, num_hidden_layers)
        # Create the decoder module
        self.decoder = Decoder(embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias, num_hidden_layers)

    def forward(self, x, enc_output):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output = self.encoder(embedding_output)
        decoder_output = self.decoder(x, encoder_output, enc_output)
        return decoder_output