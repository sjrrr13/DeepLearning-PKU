import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision

import numpy as np
import pickle
import random
import matplotlib.pyplot as plt

from utils import load_model


def load_data(path):
    transform_test = transforms.Compose([
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))
    ])

    with open(path, 'rb') as f:
        datas, labels = pickle.load(f)
    
    datas = np.array(datas) / 255
    datas = torch.from_numpy(datas).view(1000, 1, 28, 28).float()
    datas = transform_test(datas)

    labels = np.array(labels)
    labels = torch.from_numpy(labels).view(1000).long()

    test_data = torch.utils.data.TensorDataset(datas, labels)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    return test_loader


def attack_mcmc(data, label, model, device, max_step=200, std=0.03, threshold=0.1):
    attack_label = (label + 1) % 10
    
    for _ in range(max_step):
        gamma = torch.rand(1, device=device)
        data_new = torch.normal(data, std=std).to(device)
        
        with torch.no_grad():
            pred = F.softmax(model(data_new), dim=-1)
        
        y_probs = pred[0][attack_label]
        if y_probs >= gamma:
            continue
        if F.l1_loss(data_new, data, reduction='mean') > threshold:
            continue

        adv_label = model(data_new).max(1, keepdim=True)[1]
        if adv_label == attack_label:
            return True, data
    
    return False, data


def attack(model, test_loader, device):
    success = 0
    adv_samples = []

    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        assert init_pred.item() == label.item()
        
        res, adv_data = attack_mcmc(data, label, model, device, max_step=300, std=0.02, threshold=0.05)
        if res:
            adv_samples.append((data.cpu().detach().numpy().squeeze(), adv_data.cpu().detach().numpy().squeeze(), label.item()))
            success = success + 1
    
    print(f"MCMC Attack Success Ratio: {success / 1000}")
    return 100.0 * success / 1000, adv_samples
        

def show_adv_img(path, device):
    test_data = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())
    class_names = list(test_data.classes)

    with open("../black_result/mcmc-7.5.pkl") as f:
        adv_samples = pickle.load(f)
    print(adv_samples[0])
    exit()
    
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
    plt.savefig("../black_result/mcmc_result.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(device)
    model = load_model("../model/resnet34.ckpt")
    model.eval()
    test_loader = load_data('../attack_data/white_correct_1k.pkl')

    attack_rate, adv_samples = attack(model, test_loader, device)

    with open(f"../black_result/mcmc-{attack_rate}.pkl", 'wb') as f:
        pickle.dump(adv_samples, f)
