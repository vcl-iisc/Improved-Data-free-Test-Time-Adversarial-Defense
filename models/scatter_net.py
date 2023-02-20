# ref : https://github.com/fbcotter/pytorch_wavelets

from torch import nn
import torch
from pytorch_wavelets import ScatLayer


class Scatternet(nn.Module):
    def __init__(self, num_layers=3, droprate =0):
        super(Scatternet, self).__init__()
        s = []
        for i in range(num_layers):
            s.append(ScatLayer())

        self.model = torch.nn.Sequential(*s)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(droprate)
        

    def forward(self, x):
        if x.size(1) ==1:
            x= x.expand(-1,3,-1,-1)
        z = self.model(x)
        z = self.avg_pool(z)
        feature = z.view(z.size(0), -1)
        feature = self.dropout(feature)
        return feature
        #return self.linear(feature)


if __name__ == '__main__':
    x = torch.randn(32, 3, 32, 32)
    scat = Scatternet()
    z = scat(x)
    print(z.shape)