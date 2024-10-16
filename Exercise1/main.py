import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

M = 2
Q = 1
SIGMA = 0.3

"""
LINEAR FIT
"""


def linear_function(m: float, q: float, x: float) -> float:
    return m * x + q


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Training loop
def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=30):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        # Training phase
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Zero out gradients from the previous step
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item() * inputs.size(0)  # Accumulate total loss for this batch

        epoch_loss = running_loss / len(train_loader.dataset)  # Average loss for the epoch
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # No need to compute gradients for validation
            for val_inputs, val_targets in valid_loader:
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item() * val_inputs.size(0)

        val_loss = val_loss / len(valid_loader.dataset)  # Average validation loss
        print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}')


def main():
    np.random.seed(0)
    x_train = np.random.uniform(-1, 1, 500)
    x_valid = np.random.uniform(-1, 1, 50)
    x_valid.sort()
    y_target = linear_function(M, Q, x_valid)
    y_train = np.random.normal(linear_function(M, Q, x_train),
                               SIGMA)
    y_valid = np.random.normal(linear_function(M, Q, x_valid), SIGMA)

    plt.plot(x_valid, y_target, label='target')
    plt.scatter(x_valid, y_valid, color='r', label='validation data')
    plt.legend()
    plt.grid(True)
    plt.show()

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32).unsqueeze(1)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=30)


if __name__ == '__main__':
    main()
