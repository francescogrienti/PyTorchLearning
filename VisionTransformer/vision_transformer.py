import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim

# TODO Check if the notation of torch.einsum is correct, there could be problems with dimension of tensors!
# TODO Not clear the role of [CLS] token and position_embeddings, still in progress.

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


class PatchEmbedding(nn.Module):
    """
    Patches embedding module, combination of patches, position and class embeddings.
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
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, embed_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
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

        # To fix the dim = 3
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
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
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
            block = TransformerBlock(embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias)
            self.blocks.append(block)

    def forward(self, x, ):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x = block(x)
        return x


class ViTForClassification(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias, num_hidden_layers,
                 dataset, num_classes, patch_size, num_channels):
        super().__init__()
        self.image_size = dataset[0][0].shape[-1]
        self.embed_size = embed_size
        self.num_classes = num_classes
        # Create the embedding layer
        self.embedding = PatchEmbedding(patch_size, embed_size, num_channels, self.image_size, dropout)
        # Create the transformer encoder module
        self.encoder = Encoder(embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias, num_hidden_layers)
        # Create a linear layer to project the encoder's output to the number of classes: why just one linear layer in the
        # MLP classifier attached to the encoder?
        self.classifier = nn.Linear(self.embed_size, self.num_classes)
        # Initialize the weights: how are these initialized?
        self.apply(self._init_weights)

    def forward(self, x):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output = self.encoder(embedding_output)
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0])
        # Return the logits
        return logits


"""
FUNCTIONS
"""


def train_and_test_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    train_losses = [0 for _ in range(epochs)]
    test_losses = [0 for _ in range(epochs)]
    train_accuracies = [0 for _ in range(epochs)]
    test_accuracies = [0 for _ in range(epochs)]
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training phase
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Zero out gradients from the previous step
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            _, predicted = torch.max(outputs.data, 1)  # Get class with the highest probability
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()  # Accumulate total loss for this batch

        epoch_loss = running_loss / len(train_loader.dataset)  # Average loss for the epoch
        train_losses[epoch] = epoch_loss
        train_accuracies[epoch] = correct_train / total_train
        print(
            f'Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracies[epoch]:.4f}')

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():  # No need to compute gradients for validation
            for test_inputs, test_targets in test_loader:
                test_outputs = model(test_inputs)
                test_loss += criterion(test_outputs, test_targets).item()
                _, predicted = torch.max(test_outputs.data, 1)  # Get class with highest probability
                total_test += test_targets.size(0)
                correct_test += (predicted == test_targets).sum().item()

        test_loss = test_loss / len(test_loader.dataset)  # Average validation loss
        test_losses[epoch] = test_loss
        test_accuracies[epoch] = correct_test / total_test
        print(
            f'Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracies[epoch]:.4f}')

    return train_losses, test_losses, train_accuracies, test_accuracies


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

    model = ViTForClassification(EMBED_SIZE, FORWARD_EXPANSION, DROPOUT, NUM_HEADS, QKV_BIAS, NUM_HIDDEN_LAYERS,
                                 train_dataset, NUM_CLASSES, PATCH_SIZE, NUM_CHANNELS)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-2)
    train_losses, test_losses, train_accuracies, test_accuracies = train_and_test_model(model, train_loader,
                                                                                        test_loader,
                                                                                        criterion, optimizer, EPOCHS)
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(30, 12))
    ax0.plot(train_losses, label='Train Loss')
    ax0.plot(test_losses, label='Test Loss')
    ax0.set_ylabel('Model Loss')
    ax0.set_xlabel('Epoch')
    ax0.grid(True)
    ax0.legend(['Train', 'Test'], loc='best')
    ax0.set_title('Loss function')

    ax1.plot(train_accuracies, label='Training accuracy')
    ax1.plot(test_accuracies, label='Test accuracy')
    ax1.set_ylabel('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)
    ax1.legend(['Train', 'Test'], loc='best')
    ax1.set_title('Accuracy function')

    plt.savefig('ViT_loss_accuracy.png')
    plt.show()


if __name__ == '__main__':
    main()