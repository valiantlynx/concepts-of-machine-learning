import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 512)  # Adjust input size for CIFAR-10
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            images, labels = images.view(images.size(0), -1), labels
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.view(images.size(0), -1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
            
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, precision, recall, f1

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)
    cifar_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader_mnist = DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader_mnist = DataLoader(mnist_test, batch_size=64, shuffle=False)
    train_loader_cifar = DataLoader(cifar_train, batch_size=64, shuffle=True)
    test_loader_cifar = DataLoader(cifar_test, batch_size=64, shuffle=False)
    
    return train_loader_mnist, test_loader_mnist, train_loader_cifar, test_loader_cifar

# Example usage
train_loader_mnist, test_loader_mnist, train_loader_cifar, test_loader_cifar = load_data()
model_mnist = MyModel(num_classes=10)
model_cifar = MyModel(num_classes=10)

criterion = nn.CrossEntropyLoss()
optimizer_mnist = optim.Adam(model_mnist.parameters())
optimizer_cifar = optim.Adam(model_cifar.parameters())

train_model(model_mnist, train_loader_mnist, criterion, optimizer_mnist)
train_model(model_cifar, train_loader_cifar, criterion, optimizer_cifar)

accuracy_mnist, precision_mnist, recall_mnist, f1_mnist = evaluate_model(model_mnist, test_loader_mnist)
accuracy_cifar, precision_cifar, recall_cifar, f1_cifar = evaluate_model(model_cifar, test_loader_cifar)
