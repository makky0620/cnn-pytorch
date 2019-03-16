# coding: utf-8

import torch
import matplotlib.pyplot as plt
import numpy as np

from model import CNN
from util import load_data

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == "__main__":
    
    train_loader, test_loader = load_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0

    net = CNN()
    net.load_state_dict(torch.load('model_data/model1.pth'))

    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
