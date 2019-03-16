# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from model import CNN
from util import load_data

if __name__ == "__main__":
    
    train_loader, test_loader = load_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(2):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:
                print("[{}, {}] loss: {}".format(epoch+1, i+1, running_loss/2000))
                running_loss = 0.0
        
        torch.save(deepcopy(net).cpu().state_dict(), 'model_data/model'+str(epoch)+'.pth')
