import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

class ScatternetDetector(nn.Module):
    def __init__(self, input_dim=10, flat_dim=10):
        super(ScatternetDetector, self).__init__()
        self.flat_dim = flat_dim
        self.linear = torch.nn.Linear(input_dim,flat_dim)  # layer from scatter net
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = weightNorm(nn.Linear(128,2), name='weight')

        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
        self.batchnorm = nn.BatchNorm1d(128)

 
    def forward(self, x):
        
        x = self.linear(x)
        ## Corresponding to the FE part
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        ## Batch Norm Bottleneck Part
        x = F.relu(self.fc2(x))
        x = self.batchnorm(x)
        x = self.dropout(x)

        self.features_test = x
        
        ## Locked Classifier.
        x = self.fc3(x)

        return x




class Net(nn.Module):
    def __init__(self, flat_dim=10):
        super(Net, self).__init__()
        self.flat_dim = flat_dim
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = weightNorm(nn.Linear(128,2), name='weight')

        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
        self.batchnorm = nn.BatchNorm1d(128)

 
    def forward(self, x):
      
        ## Corresponding to the FE part
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        ## Batch Norm Bottleneck Part
        x = F.relu(self.fc2(x))
        x = self.batchnorm(x)
        x = self.dropout(x)

        self.features_test = x
        
        ## Locked Classifier.
        x = self.fc3(x)

        return x
