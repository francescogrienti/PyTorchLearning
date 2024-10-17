import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

M = 2
Q = 1
SIGMA = 0.3
EPOCHS = 30

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
def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs):
    train_losses = [0 for _ in range(epochs)]
    valid_losses = [0 for _ in range(epochs)]
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
        train_losses[epoch] = epoch_loss
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # No need to compute gradients for validation
            for val_inputs, val_targets in valid_loader:
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item() * val_inputs.size(0)

        val_loss = val_loss / len(valid_loader.dataset)  # Average validation loss
        valid_losses[epoch] = val_loss
        print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}')

    return train_losses, valid_losses


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
    plt.savefig('linear_fit.png')
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
    train_losses, valid_losses = train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=EPOCHS)

    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(30, 12))
    ax0.plot(train_losses, label='Train Loss')
    ax0.plot(valid_losses, label='Validation Loss')
    ax0.set_ylabel('Model Loss')
    ax0.set_xlabel('Epoch')
    ax0.legend(['Train', 'Validation'], loc='best')
    ax0.set_title('Loss -- Model 1')

    x_predicted = np.random.uniform(-1, 1, 200)  # Same as in TensorFlow
    x_predicted_tensor = torch.tensor(x_predicted, dtype=torch.float32).unsqueeze(1)  # Reshape for model input

    # Set the model to evaluation mode before making predictions
    model.eval()
    with torch.no_grad():
        y_predicted_tensor = model(x_predicted_tensor)  # Forward pass to make predictions

    y_predicted = y_predicted_tensor.squeeze().numpy()  # Convert predictions to NumPy array for plotting

    # Plot the predictions vs. validation target
    ax1.scatter(x_predicted, y_predicted, color='r', label='Predicted Data')
    ax1.plot(x_valid, y_target, label='Target')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.set_title('Prediction -- Model 1')
    ax1.grid(True)
    plt.savefig('train_valid_linear_fit.png')
    plt.show()


if __name__ == '__main__':
    main()
