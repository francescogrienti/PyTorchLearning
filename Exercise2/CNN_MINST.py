from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

"""
CNN FOR IMAGE RECOGNITION: THE USAGE OF CONVOLUTIONAL LAYERS
"""

# TorchMetrics is a collection of 100+ PyTorch metrics implementations and an easy-to-use API to create custom metrics
# https://pytorch.org/docs/stable/tensorboard.html This may be helpful!


EPOCHS = 20
HIDDEN_SIZES = [10, 20]
INPUT_CHANNELS = 1
OUTPUT_SIZE = 10
IMG_ROWS, IMG_COLS = 28, 28


# TODO the image dimensions are not a parameter when defining Conv2D, the model infers them on its own. Understand the exact meaning of out_channels.
# TODO Understand why it is not useful to define a softmax layer in the data members of the class. Apparently,
# TODO the reason resides in the fact that the crossEntropyLoss contains itself the softmax activation. In tensorflow it does not work like this.
# TODO Why this? Try to dive deeper into this.
# TODO How is the kernel function initialized by default? Kaiming Normal Initialization for the weight, zero for the bias.
# TODO the kernel is basically a dot product between a kernel matrix and input data plus a bias.
# TODO Why hidden_sizes used like this?


class CNN(nn.Module):
    def __init__(self, input_size=INPUT_CHANNELS, hidden_sizes=None, output_size=OUTPUT_SIZE, dropout_prob=0.5):
        super(CNN, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = HIDDEN_SIZES
        self.relu = nn.ReLU()
        self.fc1 = nn.Conv2d(in_channels=input_size, out_channels=hidden_sizes[0], kernel_size=7, stride=1,
                             padding=1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Conv2d(in_channels=hidden_sizes[0], out_channels=hidden_sizes[1], kernel_size=3, stride=1,
                             padding=1)
        self.fc3 = nn.MaxPool2d(kernel_size=4)
        self.flatten = nn.Flatten()
        # Calculate the size after convolutions and pooling
        # Assuming IMG_ROWS and IMG_COLS are known dimensions of input
        conv_output_size = self._get_conv_output((input_size, IMG_ROWS, IMG_COLS))
        self.output = nn.Linear(conv_output_size, output_size)

    def _get_conv_output(self, shape):
        # Helper function to determine output size after convolutions and pooling
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.fc3(self.fc2(self.fc1(x)))
            return x.numel()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.flatten(x)
        x = self.output(x)

        return x


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
            _, predicted = torch.max(outputs.data, 1)  # Get class with highest probability
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
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    model = CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
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

    plt.savefig('CNN_loss_accuracy.png')
    plt.show()


if __name__ == '__main__':
    main()
