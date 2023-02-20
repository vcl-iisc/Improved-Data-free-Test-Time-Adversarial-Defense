from torch import nn
import torch
from torchvision.models import resnet34, ResNet34_Weights
import os

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super(AdaptiveConcatPool2d, self).__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def head_blocks(in_dim, p, out_dim, activation=None):
    "Basic Linear block"
    layers = [
        nn.BatchNorm1d(in_dim),
        nn.Dropout(p),
        nn.Linear(in_dim, out_dim)
    ]
    
    if activation is not None:
        layers.append(activation)
        
    return layers       

def requires_grad(layer):
    "Determines whether 'layer' requires gradients"
    ps = list(layer.parameters())
    if not ps: return None
    return ps[0].requires_grad


def create_head(nf, nc, bn_final=False,dad_dropout=0.0):
    "Model head that takes in 'nf' features and outputs 'nc' classes"
    pool = AdaptiveConcatPool2d()
    layers = [pool, nn.Flatten()]
    layers += head_blocks(nf, 0.25, 512, nn.ReLU(inplace=True))
    layers += head_blocks(512, 0.5, nc)
    
    if bn_final:
        layers.append(nn.BatchNorm1d(nc, momentum=0.01))
    
    return nn.Sequential(*layers)

       
def cnn_model( nc=17, bn_final=True, init=nn.init.kaiming_normal_ , dad_dropout=0.0):
    "Creates a model using a pretrained 'model' and appends a new head to it with 'nc' outputs"
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

    # remove dense and freeze everything
    body = nn.Sequential(*list(model.children())[:-2])
    head = create_head(1024, nc, bn_final, dad_dropout=dad_dropout)
    
    model = nn.Sequential(body, head)

   

    # initialize the weights of the head
    for child in model[1].children():
        if isinstance(child, nn.Module) and (not isinstance(child, bn_types)) and requires_grad(child): 
            init(child.weight)
    
    return model

def enable_dad_dropout(model):
    model[1][3].train()