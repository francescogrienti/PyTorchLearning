import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import os

"""
In this .py we implement the MAE (MaskedAutoEncoder) with the aim of reconstructing masked images 
from the CIFAR Database. The general architecture of a MAE is made up of: masked input, encoder, decoder. We will 
implement the encoder and the decoder exploiting the Attention Mechanism. This work belongs to the Self-Supervised Learning 
general framework. 
"""

# System
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":4096:8" for more memory usage

# Hyperspace
hyper_space = {
    "embed_size": 36,
    "decoder_embed_size": 24,
    "num_patches": 64,
    "num_heads": 4,
    "encod_hidden_layers": 8,
    "decod_hidden_layers": 4,
    "forward_expansion": 108,
    "patch_size": 4,
    "dropout_rate": 0.1,
    "learning_rate": 0.0001,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
    "epochs": 300,
    "warmup_steps": 80,
    "mask_ratio": 0.75,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(1337)
torch.use_deterministic_algorithms(True)
g = torch.Generator()
g.manual_seed(0)

print(f"Using device: {device}", flush=True)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


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
        # The layer projects each patch into a vector of size embed_size
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

    def forward(self, x, mask_ratio):
        x = self.embedding(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        # masking: length -> length * mask_ratio
        # add pos embed w/o cls token
        x = x + self.position_embeddings[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_tokens = self.cls_token + self.position_embeddings[:, 1:, :]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        return x, mask, ids_restore

    @staticmethod
    def random_masking(x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


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


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the Encoder module.
    """

    def __init__(self, embed_size, num_attention_heads, dropout, qvk_bias):
        super().__init__()
        self.embed_size = embed_size
        self.num_attention_heads = num_attention_heads
        # The attention head size is the embed size divided by the number of attention heads
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

    def __init__(self, embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias, encod_hidden_layers):
        super().__init__()
        # Create a list of transformer blocks
        self.layer_norm = nn.LayerNorm(embed_size)
        self.blocks = nn.ModuleList([])
        for _ in range(encod_hidden_layers):
            block = TransformerBlock(embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias)
            self.blocks.append(block)

    def forward(self, x):
        # Calculate the transformer block's output for each block
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_size, decoder_embed_size, num_patches, patch_size, num_channels, forward_expansion,
                 dropout, num_attention_heads,
                 qvk_bias, decod_hidden_layers):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_size, decoder_embed_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_size))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_size),
                                              requires_grad=False)  # fixed sin-cos emb
        self.decoder_pred = nn.Linear(decoder_embed_size, patch_size ** 2 * num_channels, bias=True)  # decoder to patch
        self.layer_norm = nn.LayerNorm(decoder_embed_size)
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(decod_hidden_layers):
            block = TransformerBlock(decoder_embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias)
            self.blocks.append(block)

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # Calculate the transformer block's output for each block
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        return x


class MaskedAutoEncoder(nn.Module):
    def __init__(self, embed_size, decoder_embed_size, num_patches, forward_expansion, dropout, num_attention_heads,
                 qvk_bias, dataset, patch_size,
                 encod_hidden_layers, decod_hidden_layers, num_channels, norm_pix_loss=False):
        super().__init__()
        self.image_size = dataset[0][0].shape[-1]
        self.embed_size = embed_size
        self.decoder_embed_size = decoder_embed_size
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        # Create the embedding layer
        self.embedding = PatchEmbedding(patch_size, embed_size, num_channels, self.image_size, dropout)
        # Create the encoder module
        self.encoder = Encoder(embed_size, forward_expansion, dropout, num_attention_heads, qvk_bias,
                               encod_hidden_layers)
        # Create the decoder module
        self.decoder = Decoder(embed_size, decoder_embed_size, num_patches, patch_size, num_channels, forward_expansion,
                               dropout, num_attention_heads, qvk_bias, decod_hidden_layers)

    # Function for the patches
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % self.patch_size == 0

        h = w = imgs.shape[2] // self.patch_size
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, self.patch_size, w, self.patch_size))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, self.patch_size ** 2 * 3))
        return x

    # Function for evaluating the loss between original images and the prediction from the MAE
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, mask_ratio=0.75):
        # Calculate the embedding output
        embedding_output, mask, ids_restore = self.embedding(x, mask_ratio)
        # Calculate the encoder's output
        encoder_output = self.encoder(embedding_output)
        decoder_output = self.decoder(encoder_output, ids_restore)
        loss = self.forward_loss(x, decoder_output, mask)
        return loss, decoder_output, mask


def train_model(model, optimizer, epochs, mask_ratio, linear_warmup, cosine_lr):
    train_losses = [0 for _ in range(epochs)]
    test_losses = [0 for _ in range(epochs)]
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        # Training phase
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Zero out gradients from the previous step
            loss, _, _ = model(inputs, mask_ratio)
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()  # Accumulate total loss for this batch

        epoch_loss = running_loss / len(train_loader)  # Average loss for the epoch
        train_losses[epoch] = epoch_loss
        print(
            f'Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.4f}', flush=True)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        test_loss = 0.0
        with torch.no_grad():  # No need to compute gradients for validation
            for test_inputs, test_targets in test_loader:
                test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
                loss, _, _ = model(test_inputs, mask_ratio)
                test_loss += loss.item()

        test_loss = test_loss / len(test_loader)  # Average validation loss
        test_losses[epoch] = test_loss
        print(
            f'Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss:.4f}', flush=True)

        if epoch < hyper_space["warmup_steps"]:
            linear_warmup.step()
        else:
            cosine_lr.step()

    return train_losses, test_losses


def main():
    model = MaskedAutoEncoder(hyper_space["embed_size"], hyper_space["decoder_embed_size"], hyper_space["num_patches"],
                              hyper_space["forward_expansion"], hyper_space["dropout_rate"], hyper_space["num_heads"],
                              hyper_space["qkv_bias"], train_dataset, hyper_space["patch_size"],
                              hyper_space["encod_hidden_layers"], hyper_space["decod_hidden_layers"],
                              hyper_space["num_channels"]).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=hyper_space["learning_rate"], weight_decay=1e-2)
    linear_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, end_factor=1.0,
                                                total_iters=hyper_space["warmup_steps"], last_epoch=-1)
    cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                     T_max=hyper_space["epochs"] - hyper_space["warmup_steps"],
                                                     eta_min=1e-5)
    train_losses, test_losses = train_model(model, optimizer, hyper_space["epochs"], hyper_space["mask_ratio"],
                                            linear_warmup, cos_decay)
    # Plot Loss
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.ylabel('Model Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(['Train', 'Test'], loc='best')
    plt.title('Loss function - Masked Autoencoder with LR adaptation')
    # Convert dictionary to table format
    table_data = [[k, v] for k, v in hyper_space.items()]
    table = plt.table(cellText=table_data, colLabels=["Hyperparameter", "Value"],
                      cellLoc='center', loc='upper right', bbox=[1.05, 0, 0.4, 0.3])

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig('MAE_loss.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
