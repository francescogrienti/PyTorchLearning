import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

SIGMA = 0.5
EPOCHS = 50
HIDDEN_SIZES = [30, 20, 10]

"""
TWO-DIMENSIONAL FIT
"""


def sin(x, y):
    return np.sin(x ** 2 + y ** 2)


class ExtendedNN(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=None, output_size=1):
        super(ExtendedNN, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = HIDDEN_SIZES
        self.relu = nn.ReLU()  # Define RELU as a layer
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.output = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.output(x)
        return x


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
    x_train = np.random.uniform(-1.5, 1.5, size=(10000, 2))
    x_valid = np.random.uniform(-1.5, 1.5, size=(1000, 2))
    x_valid.sort()
    y_target = sin(x_valid[:, 0], x_valid[:, 1])
    y_train = np.random.normal(sin(x_train[:, 0], x_train[:, 1]), SIGMA)
    y_valid = np.random.normal(sin(x_valid[:, 0], x_valid[:, 1]), SIGMA)

    test_points = np.random.uniform(-1.5, 1.5, size=(2, 50))
    test_points.sort()
    x, y = np.meshgrid(test_points[0], test_points[1])
    z_target = sin(x, y)
    z_validation = np.random.normal(sin(x, y), SIGMA)

    fig = plt.figure(figsize=(18, 13))
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z_validation, color='green', linewidth=0.1)
    ax.contour3D(x, y, z_target, 30, cmap='plasma')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Training data', fontsize=20)
    plt.grid(True)
    plt.savefig('2_dimensional_fit.png')
    plt.show()

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    model = ExtendedNN()
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

    x_predicted = np.random.uniform(-1.5, 1.5, size=(1000, 2))  # Same as in TensorFlow
    x_predicted_tensor = torch.tensor(x_predicted, dtype=torch.float32)

    # Set the model to evaluation mode before making predictions
    model.eval()
    with torch.no_grad():
        y_predicted_tensor = model(x_predicted_tensor)  # Forward pass to make predictions

    y_predicted = y_predicted_tensor.squeeze().numpy()  # Convert predictions to NumPy array for plotting

    test_points = np.random.uniform(-1.5, 1.5, size=(2, 50))
    test_points.sort()
    x, y = np.meshgrid(test_points[0], test_points[1])
    z_target = sin(x, y)

    fig = plt.figure(figsize=(18, 13))
    ax = plt.axes(projection='3d')
    ax.scatter(x_predicted[:, 0], x_predicted[:, 1], y_predicted, color='green', linewidth=0.1, label='predicted')
    ax.contour3D(x, y, z_target, 30, cmap='plasma')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    ax.set_title('Predicted data', fontsize=20)
    plt.grid(True)
    plt.savefig('prediction_2D.png')
    plt.show()


if __name__ == '__main__':
    main()
