import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp


def train(model, Xtrain_, Ytrain_,  optimizer, device):
    
    for epoch in range(1, 1001):
        
        model.train()
        pid = os.getpid()

        optimizer.zero_grad()
        Ypred = model(Xtrain_.to(device))

        loss = nn.NLLLoss()(Ypred , Ytrain_.to(device))
        loss.backward()

        optimizer.step()
        
        if epoch % 100 == 0:
                print('{}\tTrain Epoch: {} \tLoss: {:.6f}'.format(
                    pid, epoch, loss.item()))