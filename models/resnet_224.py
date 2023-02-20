import copy
import torch
from torch  import nn
from torchvision.models import resnet50, ResNet50_Weights

from dataset.tiny_imagenet import TinyImagenet
class Resnet50(nn.Module):
    def __init__(self,num_channels=3 , num_classes=200, dad_dropout = 0.0):
        super(Resnet50, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        self.dad_dropout = torch.nn.Dropout(p=dad_dropout)
        self.fc = nn.Linear(2048, num_classes)
        for param in self.model.parameters():
            param.requires_grad = False
        

        
    
    def forward(self, x,out_feature=False):
        x = self.model(x)
        feature = self.dad_dropout(x)
        x = self.fc(feature)
        if out_feature == False:
            return x
        else:
            return x,feature
        
import torch
from torch  import nn
from torchvision.models import resnet34, ResNet34_Weights
class Resnet34(nn.Module):
    def __init__(self,num_channels=3 , num_classes=200, dad_dropout = 0.0):
        super(Resnet34, self).__init__()
        self.model = resnet34(False, num_classes=num_classes)
        self.dad_dropout = nn.Dropout(dad_dropout)
    
    def forward(self, x,out_feature=False):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        feature = self.dad_dropout(x)
        x = self.model.fc(x)
        if out_feature == False:
            return x
        else:
            return x,feature
        
    def load_state_dict(self, weights):
        self.model.load_state_dict(weights)




if __name__ == "__main__":
    model = Resnet34()
    model.load_state_dict(torch.load("/media2/inder/GenericRobustness/black-box-ripper/checkpoints/teacher_resnet34_tiny_imagenet_for_tiny_imagenet_state_dict", map_location="cpu"))
    import evaluate_model
    dataset = TinyImagenet()
    
    evaluate_model.evaluate(model,dataset.test_dataloader())