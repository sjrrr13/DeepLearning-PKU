import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet34
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import random

from utils import load_model


def fgsm_attack(data, epsilon, max_iter, model, label):
    assert epsilon > 0, "epsilon must be positive"
    x_adv = data
    attack_label = (label + 1) % 10
    criterion = nn.CrossEntropyLoss()

    for _ in range(max_iter):
        x_adv.requires_grad_(True)
        x_adv.retain_grad()
        output = model(x_adv)
        loss = criterion(output, attack_label)
        
        model.zero_grad()
        loss.backward(retain_graph=True)
        if x_adv.grad is None:
            return False, x_adv
        
        data_grad = x_adv.grad.data
        x_adv = x_adv - epsilon * data_grad.sign()

        x_adv = torch.clamp(x_adv, 0, 1)
        if model(x_adv).max(1, keepdim=True)[1] == attack_label:
            return True, x_adv
    
    return False, x_adv

def attack(model, device, test_loader, epsilon):
    model.eval()
    success = 0
    adv_samples = []
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        success, perturbed_data = fgsm_attack(image=data, epsilon=epsilon, max_iter=100, model=model, target=target)
        if success:
            success += 1
            adv_samples.append((data, perturbed_data, target))

    attack_acc = 100.0 * success / 1000
    print(f"Epsilon: {epsilon}\tAttack success rate: {attack_acc:.2f}")
    return attack_acc, adv_samples


def prepare_attack_sample():
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])
    train_data = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    model = load_model("../model/resnet34.ckpt")
    
    attack_accs = []
    samples = []

    acc, sample = attack(model, device, train_loader, epsilons=0.05)
    attack_accs.append(acc)
    samples.append(sample)
    torch.save(sample, f'../attack_data/White-Train.pt')


def add_sample_to_train(path, device):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3529,)),
        transforms.RandomErasing()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3529,))
    ])

    _train_dataset = datasets.FashionMNIST(root='../data/', download=False, train=True, transform=transform_train)
    test_dataset = datasets.FashionMNIST(root='../data/', download=False, train=False, transform=transform_test)

    avc_samples = torch.load(path, map_location=device)
    random.seed(2024)
    random.shuffle(avc_samples)
    
    train_imgs=[]
    train_labels=[]
    cnt = 0

    for _, perturbed_data, target in avc_samples:
        train_imgs.append(perturbed_data.cpu().detach().numpy().squeeze())
        train_labels.append(target.cpu().detach().numpy().squeeze())
        cnt += 1
        
    for i in range(len(_train_dataset)):
        train_imgs.append(_train_dataset[i][0].cpu().detach().numpy().squeeze())
        train_labels.append(_train_dataset[i][1])
        cnt += 1
    
    train_imgs = np.array(train_imgs)
    train_imgs = torch.from_numpy(train_imgs).reshape(cnt, 1, 28, 28).float()
    train_labels = np.array(train_labels)
    train_labels = torch.from_numpy(train_labels).long()
    
    train_dataset = torch.utils.data.TensorDataset(train_imgs, train_labels)
    
    return train_dataset, test_dataset


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(device)

    # Prepare white box attack samples at first
    # prepare_attack_sample()

    train_dataset, test_dataset = add_sample_to_train("../attack_data/White-Train.pt", device)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=8)

    test_accuracies, train_accuracies = [], []

    # New classifier
    model = resnet34()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) 
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    acc_max = 90
    for epoch in range(50):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100. * correct / total

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100.0 * correct / total
        
        scheduler.step()

        if test_accuracy > acc_max:
            acc_max = test_accuracy
            torch.save(model.state_dict(), f"../checkpoints/ResNet34-{test_accuracy}_new.ckpt")
            print(f"Save model at epoch {epoch} with accuracy: {test_accuracy}%")

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    
    plt.plot(train_accuracies, label='Train')
    plt.plot(test_accuracies, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig('../checkpoints/Accuracy_new.png')
        