import torch
import torchvision
from torchvision import transforms
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from utils import load_model
import pickle
import random


def load_data():
    transform_test = transforms.Compose([
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))
    ])

    with open('../attack_data/white_correct_1k.pkl', 'rb') as f:
        datas, labels = pickle.load(f)
    
    datas = np.array(datas) / 255
    datas = torch.from_numpy(datas).view(1000, 1, 28, 28).float()
    datas = transform_test(datas)

    labels = np.array(labels)
    labels = torch.from_numpy(labels).view(1000).long()

    test_data = torch.utils.data.TensorDataset(datas, labels)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    return test_loader


def fgsm_attack(model, data, label, epsilon, max_iter):
    x_adv = None
    assert epsilon > 0, "epsilon must be positive"
    
    x_adv = data.view(-1, 1, 28, 28)
    attack_label = (label + 1) % 10
    criterion = nn.CrossEntropyLoss()

    for _ in range(max_iter):
        # Compute gradients
        x_adv.requires_grad_(True)
        x_adv.retain_grad()
        output = model(x_adv)
        loss = criterion(output, attack_label)
        # Backward to update `x_adv`
        model.zero_grad()
        loss.backward(retain_graph=True)
        if x_adv.grad is None:
            return False, x_adv
        
        data_grad = x_adv.grad.data
        x_adv = x_adv - epsilon * data_grad.sign()
        # Restrict pixels to[0, 1]
        x_adv = torch.clamp(x_adv, 0, 1)
        # Attack success
        if model(x_adv).max(1, keepdim=True)[1] == attack_label:
            return True, x_adv
    return False, x_adv


def attack(model, dataloader, epsilon, device):
    model.eval()
    success = 0
    adv_samples = []
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        success, attacked_data = fgsm_attack(model, x, y, epsilon, max_iter=10)
        if success:
            success += 1
            adv_samples.append((x, attacked_data, y))
    
    attack_rate = 100.0 * success / 1000
    print(f"Epsilon: {epsilon}\tAttack success rate: {attack_rate:.2f}")
    return attack_rate, adv_samples


def show_adv_img(path, device):
    test_data = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())
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
    plt.savefig("../white_result/white_box_attack_result.png")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(device)

    epsilons = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    attack_rates = []
    samples = []

    model = load_model("../model/resnet34.ckpt")
    test_loader = load_data()

    for eps in epsilons:
        acc, sample = attack(model, test_loader, eps, device)
        attack_rates.append(acc)
        samples.append(sample)
        torch.save(sample, f'../white_result/{eps}-{acc:.2f}.pt')

    torch.save(samples, '../white_result/attack_samples.pt')

    # plt.plot(epsilons, attack_rates, "*-")
    # plt.title("I-FGSM Attack")
    # plt.xlabel("Epsilon")
    # plt.ylabel("Attack Success Rate")
    # plt.savefig("../white_result/white_attack.png")
    
    # show_adv_img("../white_result/0.05-84.30.pt", device)
