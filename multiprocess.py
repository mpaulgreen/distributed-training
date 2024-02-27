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
from train import train

input_size = 13
output_size = 3
hidden_size = 100

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        X = torch.sigmoid((self.fc1(X)))
        X = torch.sigmoid(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=-1)
    


if __name__ == '__main__':
    
    device = torch.device("cuda")
    wine_data = pd.read_csv('data/wine_data.csv')
    wine_features = wine_data.drop('Class', axis = 1)
    wine_target = wine_data[['Class']]
    X_train, x_test, Y_train, y_test = train_test_split(wine_features,
                                                    wine_target,
                                                    test_size=0.4,
                                                    random_state=0)
    
    
    Xtrain_ = torch.from_numpy(X_train.values).float()
    Xtest_ = torch.from_numpy(x_test.values).float()
    Ytrain_ = torch.from_numpy(Y_train.values).view(1,-1)[0]
    Ytest_ = torch.from_numpy(y_test.values).view(1,-1)[0]
    
    mp.set_start_method('spawn', force=True)
    Xtrain_.share_memory_()
    Ytrain_.share_memory_() 
    Xtest_.share_memory_()
    Ytest_.share_memory_()

    model = Net().to(device)
    model.share_memory() # gradients are allocated lazily, so they are not shared here
    
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    processes = []
    for rank in range(4):
        
        p = mp.Process(target=train, args=(model, Xtrain_, Ytrain_, optimizer,device ))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    if(processes[0].is_alive()==False):
        
        model.eval()
        test_loss = 0
        correct = 0

        predict_out = model(Xtest_.to(device))
        _, predict_y = torch.max(predict_out, 1)
        
        print("\n")
        print ('prediction accuracy', accuracy_score(Ytest_.data.cpu(), predict_y.data.cpu()))
        print ('micro precision', precision_score(Ytest_.data.cpu(), predict_y.data.cpu(), average='micro'))
        print ('micro recall', recall_score(Ytest_.data.cpu(), predict_y.data.cpu(), average='micro'))