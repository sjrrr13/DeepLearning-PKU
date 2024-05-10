import torch
import torch.nn as nn
from torchvision.models import resnet34
from torchvision import transforms

import pickle
import numpy as np

from fmnist_dataset import load_fashion_mnist


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


def load_model(path):
    """
    Load model from path.
    """
    model = resnet34()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    dict = torch.load(path)
    model.load_state_dict(dict)
    model.cuda()
    return model
