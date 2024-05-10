import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet34
import matplotlib.pyplot as plt
import os


def load_data(train_bsz, test_bsz):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,)),
        transforms.RandomErasing()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))
    ])

    train_dataset = datasets.FashionMNIST(root='../data/', download=False, train=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True, num_workers=8)
    test_dataset = datasets.FashionMNIST(root='../data/', download=False, train=False, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=test_bsz, shuffle=False, num_workers=8)
    return train_loader, test_loader


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100.0 * correct / total
    return train_loss, train_accuracy


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += criterion(output, label).item() * data.size(0)
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / total
    return test_loss, test_accuracy


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Resnet34 for 10 classes with input size 28x28
    model = resnet34()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4, eps=1e-8, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    train_loader, test_loader = load_data(train_bsz=128, test_bsz=100)

    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    acc_max = 92.0

    for epoch in range(50):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Test Acc {test_accuracy:.2f}%")
        
        if not os.path.exists('../checkpoints'):
            os.makedirs('../checkpoints/')
        if test_accuracy > acc_max:
            acc_max = test_accuracy
            torch.save(model.state_dict(), f"../checkpoints/ResNet34-{test_accuracy}.ckpt")
            print(f"Save model at epoch {epoch} with acc: {test_accuracy}%")

        train_losses.append(train_loss)
        train_accs.append(train_accuracy)
        test_losses.append(test_loss)
        test_accs.append(test_accuracy)

    # # 绘制精度曲线
    # plt.plot(train_accs, label='Train')
    # plt.plot(test_accs, label='Test')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy in Train')
    # plt.legend()
    # plt.savefig("../checkpoints/Accuracy.png")

    # # 绘制损失曲线
    # plt.plot(train_losses, label='Train')
    # plt.plot(test_losses, label='Test')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss in Train')
    # plt.title('Loss vs Epoch')
    # plt.legend()
    # plt.savefig("../checkpoints/Loss.png")
