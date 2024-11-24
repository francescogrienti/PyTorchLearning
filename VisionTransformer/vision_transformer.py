import torch
import torch.nn as nn
from torchvision import datasets, transforms


# TODO Check if the notation of torch.einsum is correct, there could be problems with dimension of tensors!
class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.
    """

    def __init__(self, embed_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.embed = embed_size
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
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, value])
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
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
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.embed_size,
                self.attention_head_size,
                self.dropout,
                self.qkv_bias
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the embedding size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return attention_output, None
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return attention_output, attention_probs


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


class TransformerBlock(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_attention_heads, dropout, qvk_bias)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, forward_expansion, dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_size)

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = \
            self.attention(self.layer_norm_1(x), output_attentions=output_attentions)
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layer_norm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return x, None
        else:
            return x, attention_probs


class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias, num_hidden_layers):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(num_hidden_layers):
            block = TransformerBlock(embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return x, None
        else:
            return x, all_attentions


# TODO Check the code more than once!
def patch_embedding(patch_size, dataset, embed_dim):
    """
    :param patch_size: the size of the patch
    :param dataset: dataset to which the patch embedding is applied
    :param embed_dim: dimension of the linear projection of the patches
    :return: Tensor containing embedded patches for the full dataset
    """

    C, H, W = dataset[0][0].shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size"
    num_patches = (H // patch_size) * (W // patch_size)
    patch_embed = nn.Linear(patch_size * patch_size * C, embed_dim)
    position_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
    all_embedded_patches = []
    for image, _ in dataset:  # Assuming dataset returns (image, label)
        patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()  # (H_p, W_p, C, P, P)
        patches = patches.view(-1, patch_size * patch_size * C)  # Flatten each patch
        embedded_patches = patch_embed(patches)  # Shape: (num_patches, embed_dim)
        patches_with_position = embedded_patches + position_embedding.squeeze(0)
        all_embedded_patches.append(patches_with_position)
    all_embedded_patches = torch.stack(all_embedded_patches)

    return all_embedded_patches


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


if __name__ == '__main__':
    main()
