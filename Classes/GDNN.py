import torch.nn as nn
import torch
from torch import optim
from sklearn import preprocessing
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc3 = nn.Linear(3, 1)

        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc3(x)) 
        return x

    def loadModel(self, path):
        self.load_state_dict(torch.load(path))
        print(self.eval())

