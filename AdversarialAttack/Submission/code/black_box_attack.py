import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from model import CNN
from utils import load_model
import random


def load_data(path):
    transform_test = transforms.Compose([
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])
    
    with open(path, 'rb') as f:
        datas, labels=pickle.load(f)
    
    datas = np.array(datas) / 255
    datas = torch.from_numpy(datas).view(1000, 1, 28, 28).float()
    datas = transform_test(datas)
    
    labels = np.array(labels)
    labels = torch.from_numpy(labels).view(1000).long()
    
    test_data = torch.utils.data.TensorDataset(datas, labels)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    return test_loader
    

# 定义攻击函数
def fgsm_attack(data, epsilon, model, attack_model, label, max_iter):
    assert epsilon > 0, 'epsilon must be positive'
    
    x_adv = data
    attack_target = ((label + 1) % 10)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(max_iter):
        x_adv.requires_grad_(True)
        x_adv.retain_grad()
        output = model(x_adv)
        loss = criterion(output, attack_target)
        
        model.zero_grad()
        loss.backward(retain_graph=True)
        if x_adv.grad is None:
            return False, x_adv
        
        data_grad = x_adv.grad.data
        x_adv = x_adv - epsilon * data_grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)
        
        with torch.no_grad():
            attack_output = attack_model((x_adv.detach() * 0.3530 + 0.2860) * 255)
            if attack_output.max(1, keepdim=True)[1] == attack_target:
                return True, x_adv

    return False, x_adv

# 定义测试函数
def attack(model, attack_model, device, test_loader, epsilon):
    model.eval()
    attack_model.eval()
    
    success = 0
    adv_samples = []
    
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)

        success, perturbed_data = fgsm_attack(data, epsilon, model, attack_model, label, max_iter=50)
        if success:
            success += 1
            adv_samples.append((data, perturbed_data, label))

    attack_acc = 100. * success / 1000
    
    print(f"Epsilon: {epsilon}\tAttack success rate: {attack_acc:.4f}")
    return attack_acc, adv_samples


def show_adv_img(path, device):
    test_data = torchvision.datasets.FashionMNIST(root='../data', train=False, download=False, transform=transforms.ToTensor())
    class_names = list(test_data.classes)

    adv_samples = torch.load(path, map_location=device)
    random.seed(2024)
    random.shuffle(adv_samples)

    torch.manual_seed(2024)
    fig = plt.figure(figsize=(9, 9))
    rows, cols = 5, 4
    cnt = 1
    for data, adv_data, label in adv_samples:
        if cnt > rows * cols:
            break
        random_idx = label.item()
        fig.add_subplot(rows, cols, cnt)
        plt.imshow(data.cpu().detach().numpy().squeeze(), cmap="gray")
        plt.title(class_names[random_idx])
        plt.axis(False)
        cnt += 1

        fig.add_subplot(rows, cols, cnt)
        plt.imshow(adv_data.cpu().detach().numpy().squeeze(), cmap="gray")
        attack_label = (random_idx + 1) % 10
        plt.title(class_names[attack_label])
        plt.axis(False)
        cnt += 1
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.savefig("../black_result/black_box_attack_result.png")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(device)

    # White box model
    model = load_model("../model/resnet34.ckpt")
    # Black box model
    attack_model = CNN()
    attack_model.load_state_dict(torch.load('../model/cnn.ckpt', map_location=device))
    attack_model.to(device)

    test_loader = load_data('../attack_data/correct_1k.pkl')
    epsilons = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
    attack_accs = []
    samples = []

    for eps in epsilons:
        acc, sample = attack(model, attack_model, device, test_loader, eps)
        attack_accs.append(acc)
        samples.append(sample)
        if not os.path.exists('../black_result'):
            os.makedirs('../black_result')    
        torch.save(sample, f'../black_result/{eps}-{acc:.2f}.pt')

    torch.save(samples, '../black_result/attack_examples.pt')

    # plt.plot(epsilons, attack_accs, "*-")
    # plt.title("Sample Migration Attack")
    # plt.xlabel("Epsilon")
    # plt.ylabel("Attack Success Rate")
    # plt.savefig('../black_result/black_attack_result.png')
    
    # Show source images and attacked images
    # show_adv_img("../black_result/0.5-0.302.pt", device)
    