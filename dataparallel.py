import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        
        print("\nInside  the model: Device id - ", torch.cuda.current_device())

        return F.log_softmax(X, dim=-1)
    
if __name__ == '__main__':
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
    Xtrain_ = Xtrain_.to(device)
    Ytrain_ = Ytrain_.to(device)
    Xtest_ = Xtest_.to(device)
    Ytest_ = Ytest_.to(device)
    model = Net()    
    model = nn.DataParallel(model) # replicated the model on each device to work on a subset of the input, chunks input into batches
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.NLLLoss()
    epochs = 100

    for epoch in range(epochs):

        optimizer.zero_grad()

        Ypred = model(Xtrain_)
        loss = loss_fn(Ypred, Ytrain_ )

        loss.backward() # in the backward pass gradients from each replica are summed into the original model
        optimizer.step()

        print("\nOutside the model: Device id - %d, Loss - %f" %(torch.cuda.current_device(), loss.item()))
        
    predict_out = model(Xtest_)
    _, predict_y = torch.max(predict_out, 1)
    
    print ('prediction accuracy', accuracy_score(Ytest_.cpu().data, predict_y.cpu().data))
    print ('micro precision', precision_score(Ytest_.cpu().data, predict_y.cpu().data, average='micro'))
    print ('micro recall', recall_score(Ytest_.cpu().data, predict_y.cpu().data, average='micro'))