# coding: utf-8

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

batch_size = 100

def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

    return train_loader, test_loader

if __name__ == "__main__":
    train, test = load_data()
    print(train)
    print(test)