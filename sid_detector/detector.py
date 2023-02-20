import torch.nn as nn
import torch
from torchvision.transforms import transforms

if __name__ == '__main__':
    import sys
    sys.path.insert(0,'..')

from .Models import Detector, FDResNet34, ResNet34, DualCIFAR4

class CombinedDetector(nn.Module):
    def __init__(self ):
        super(CombinedDetector, self).__init__()
        model = ResNet34(num_c=10)
        #m, s = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        m , s = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        normalization = transforms.Normalize(m, s)
        
        self.pd = nn.Sequential(normalization, model)
        
        model = FDResNet34(num_c=10,in_C=12)
        self.fd = nn.Sequential(normalization, model)

        #self.sid = DualCIFAR4(C_Number=3, num_class=10)
        self.sid = Detector()
    
    def load_state_dict(self,state_dicts):
        self.pd[1].load_state_dict(state_dicts['pd'])
        self.fd[1].load_state_dict(state_dicts['fd'])
        self.sid.load_state_dict(state_dicts['sid'])

    def forward(self, data):
        pd_out = self.pd(data)
        fd_out = self.fd(data)
        sid_out = self.sid(pd_out,fd_out)
        # sid out is of dimension n*3, change it to n*2 by adding last two values
        #apply softmax on sid_out
        sid_out = torch.softmax(sid_out,dim=1)
        # add last two values for each row
        sid_out[:,1] = torch.max( sid_out[:,1] , sid_out[:,2] )
        sid_out = sid_out[:,:2]
        return sid_out

if __name__ == '__main__':

    net = Detector()
    """pd_checkpoint = torch.load("/media2/inder/dad_shubham/SID/pre_trained/PDresnet_cifar10.pth" , map_location = 'cpu' )
    fd_checkpoint = torch.load("/media2/inder/dad_shubham/SID/pre_trained/FDresnet_cifar10.pth" , map_location = 'cpu' )
    sid_checkpoint= torch.load("/media2/inder/dad_shubham/SID/ExperimentRecord/KnownAttack/resnet_cifar10_BIM_0.41/checkpoint.tar", map_location='cpu')['state_dict']
    net.load_state_dict({'pd':pd_checkpoint,'fd':fd_checkpoint,'sid':sid_checkpoint})"""

    state_dict = torch.load("results/CIFAR10/known_attack_results/resnet34/PGD/detector.pt", map_location='cpu')
    net.load_state_dict(state_dict)


    net.eval()
   
    




