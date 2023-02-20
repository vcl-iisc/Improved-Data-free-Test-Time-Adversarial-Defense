import logging
import os
import random
from collections import OrderedDict

import numpy
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import TensorDataset, dataset
from torchvision import transforms

import models.resnet50 as rersnet50
import models.zskt_wresnet as zskt_wresnet
from create_adv_data import (create_adv_data_classifier,
                             create_adv_data_classifier_detector,
                             create_adv_data_detector,
                             create_adv_data_detector_classifier)
from data_list import ImageList
from data_load import mnist, usps
from dataset.cub200 import Cub2011
from dataset.oxford_pet import Databasket
from dataset.tiny_imagenet import TinyImagenet
from models import (dine_office_home_network, resnet_224, shot_networks,
                    shot_office_network, wrn)
from models.detector import Net
from models.oxford_pet_model import cnn_model
from models.resnet import ResNet18, ResNet34
from models.resnet_source import resnet18
from models.s_wrn import Network


def load_adv_data(path):
    adv_images, adv_labels = torch.load(path)
    adv_data = TensorDataset(adv_images.float()/255, adv_labels)
    
    return adv_data

class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        if len(self.mean)>1:
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
        else:
            mean, std = self.mean, self.std
        return (input - mean) / std

def load_shot_data_digit(args):

    if args.model_name == 's2m':
        train_data = mnist.MNIST_idx('./clean_data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    
                ])) 

        test_data = mnist.MNIST('./clean_data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    
                ]))

    elif args.model_name == 'u2m':
        train_data = mnist.MNIST_idx('./clean_data/mnist/', train=True, download=True,
        transform=transforms.Compose([
                    transforms.ToTensor()])) 

        test_data = mnist.MNIST('./clean_data/mnist/', train=False, download=True,
        transform=transforms.Compose([
                    transforms.ToTensor()]))

    elif args.model_name == 'm2u':

        train_data = usps.USPS_idx('./clean_data/usps/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    
                ]))

        test_data = usps.USPS('./clean_data/usps/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    
                ]))
        
    return train_data, test_data

def load_shot_data_office(args):
    
    ## prepare data
    transform =   transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    dsets = {}
    #target and test set both use the same txt file.
    txt_tar = open(args.dataset_path).readlines()
    txt_test = open(args.dataset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList(txt_tar, transform=transform)  
    dsets["test"] = ImageList(txt_test, transform=transform)

    return dsets["target"], dsets["test"]

def load_dine_data_office_home(args):

        ## prepare data
    transform =   transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    dsets = {}
    
    txt_test = open(args.dataset_path).readlines()
    
    dsets["test"] = ImageList(txt_test, transform=transform)
    
    return None, dsets["test"]


def load_zskt_model(args):
    if args.model_name =="wideresnet_16_1_source":
        config = {'input_shape':(1, 1, 32, 32), 'n_classes':args.num_classes, 'base_channels':16, 'widening_factor':1,\
            'drop_rate':0, 'depth':16}
        base_model = Network(config)
        base_model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    elif args.model_name =="wideresnet_40_2":
        base_model = wrn.WideResNet(depth=40, num_classes= args.num_classes , widen_factor=2, dropRate=0.0, dad_droprate=0)
        base_model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    else:
        base_model = zskt_wresnet.WideResNet(depth=16, num_classes=args.num_classes, widen_factor=1, dropRate=0.0, dad_droprate=args.droprate)
        base_model.load_state_dict(torch.load(args.model_path, map_location="cpu")["student_state_dict"])
    return base_model


def load_dataset(args):

    dataroot = "clean_data/"+args.dataset

    if args.method =="shot_digit":
        return load_shot_data_digit(args)
    if args.method in  ["shot_office" , "dine_office", "decision_office"]:
        return load_shot_data_office(args)
    if args.method in ["dine_office_home", "decision_office_home"]:
        return load_dine_data_office_home(args)
    
    preprocess = transforms.Compose([#transforms.Resize((32,32)),
                                        transforms.ToTensor()])
     ## Init the data
    if args.dataset == 'cifar10':
        print('Loading Cifar')
        train_data = torchvision.datasets.CIFAR10(dataroot, train=True, transform=preprocess)
        test_data = torchvision.datasets.CIFAR10(root=dataroot, train=False, transform=preprocess)
    elif args.dataset == 'mnist':
        print('Loading MNIST')
        train_data = torchvision.datasets.MNIST(dataroot, train=True, transform=preprocess, download=True)
        test_data = torchvision.datasets.MNIST(root=dataroot, train=False, transform=preprocess, download=True)
    elif args.dataset == 'fmnist':
        print('Loading FMNIST')
        train_data = torchvision.datasets.FashionMNIST(dataroot, train=True, transform=preprocess, download=True)
        test_data = torchvision.datasets.FashionMNIST(root=dataroot, train=False, transform=preprocess, download=True)
    elif  args.dataset=='svhn':
        print('Loading svhn')
        train_data = torchvision.datasets.SVHN(dataroot, split='train', transform=preprocess, download=True)
        test_data = torchvision.datasets.SVHN(root=dataroot, split='test', transform=preprocess, download=True)
    elif args.dataset=="stl":
        print('Loading stl')
        preprocess = transforms.Compose([transforms.Resize((96,96)),
                                        transforms.ToTensor()])
        train_data = torchvision.datasets.STL10(dataroot, split='train', transform=preprocess, download=True)
        test_data = torchvision.datasets.STL10(root=dataroot, split='test', transform=preprocess, download=True)
    elif args.dataset=="cifar100":
        train_data = torchvision.datasets.CIFAR100(dataroot,transform=preprocess)
        test_data = torchvision.datasets.CIFAR100(dataroot,train=False, transform=preprocess)
    elif args.dataset == "tiny_imagenet":
        test_data = TinyImagenet().test_dataset_without_normalization()
        return  None, test_data
    elif args.dataset == "oxford_pet":
        dataset = Databasket()
        train_data = dataset.train_ds 
        test_data = dataset.val_ds
    elif args.dataset == "usps":
        train_data = USPS(dataroot, train=True, transform=preprocess, download=True)
        test_data = USPS(root=dataroot, train=False, transform=preprocess, download=True)
    elif args.dataset == "cub":
        test_transform = transforms.Compose([
            transforms.Resize(256),
            #center crop,
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        test_data = Cub2011(root=dataroot,train=False, transform=test_transform, download=True)
        train_data= None
    else:
        print(f'{dataroot} / {dataset} doesnt exist')

    
    return train_data , test_data
    

def load_detector(args, load_checkpoint=False):    
    detector = Net()
        
    if load_checkpoint:
        ckpt = torch.load(args.detector_path, map_location="cpu")
        print(f'Loading Detector @ {args.detector_path}')
        #print(f'Detector -> Acc : {ckpt["acc"]} | Loss : {ckpt["loss"]}')
        detector.load_state_dict(ckpt['detector_state_dict'])
        #if num of classes is different, change the first layer
        if args.num_classes != 10:
            detector.fc1 = nn.Linear(args.num_classes,128)

    return detector


def load_shot_digit_model(args):

    mode = args.model_name

    if mode =="u2m":
        base_model = shot_networks.LeNetBase()
    elif mode=="m2u":
        base_model = shot_networks.LeNetBase()
    elif mode=="s2m":
        base_model = shot_networks.DTNBase()
    
    netB = shot_networks.feat_bootleneck(type=args.classifier, feature_dim=base_model.in_features, bottleneck_dim=args.bottleneck)
    netC = shot_networks.feat_classifier(type=args.layer, class_num = args.num_classes, bottleneck_dim=args.bottleneck, p=args.droprate)

    dir_name = os.path.dirname(args.model_path)

    base_model.load_state_dict(torch.load(os.path.join(dir_name,"target_F_par_0.1.pt"), map_location="cpu"))
    netB.load_state_dict(torch.load(os.path.join(dir_name,"target_B_par_0.1.pt"), map_location="cpu"))
    netC.load_state_dict(torch.load(os.path.join(dir_name,"target_C_par_0.1.pt"), map_location="cpu"))

    return nn.Sequential(base_model, netB, netC)    


def load_shot_office_model(args):
    
    
    if args.model_name[0:3] == 'res':
        base_model = shot_office_network.ResBase(res_name=args.model_name)
    elif args.model_name[0:3] == 'vgg':
        base_model = shot_office_network.VGGBase(vgg_name=args.model_name)

    netB = shot_office_network.feat_bootleneck(type=args.classifier, feature_dim=base_model.in_features, bottleneck_dim=args.bottleneck)
    netC = shot_office_network.feat_classifier(type=args.layer, class_num = args.num_classes, bottleneck_dim=args.bottleneck, p=args.droprate)
    
    dir_name = os.path.dirname(args.model_path)
   

    base_model.load_state_dict(torch.load(os.path.join(dir_name,"target_F_par_0.3.pt"), map_location="cpu"))
    netB.load_state_dict(torch.load(os.path.join(dir_name,"target_B_par_0.3.pt"), map_location="cpu"))
    netC.load_state_dict(torch.load(os.path.join(dir_name,"target_C_par_0.3.pt"), map_location="cpu"))

    return nn.Sequential(base_model, netB, netC)  

def load_dine_office_home_model(args):
    
    
    if args.model_name[0:3] == 'res':
        base_model = dine_office_home_network.ResBase(res_name=args.model_name)
    elif args.model_name[0:3] == 'vgg':
        base_model = dine_office_home_network.VGGBase(vgg_name=args.model_name)

    netB = dine_office_home_network.feat_bootleneck(type=args.classifier, feature_dim=base_model.in_features, bottleneck_dim=args.bottleneck)
    netC = dine_office_home_network.feat_classifier(type=args.layer, class_num = args.num_classes, bottleneck_dim=args.bottleneck, p=args.droprate)
    
    dir_name = os.path.dirname(args.model_path)
   
    base_model.load_state_dict(torch.load(os.path.join(dir_name,"finetuned_source_F.pt"), map_location="cpu"))
    netB.load_state_dict(torch.load(os.path.join(dir_name,"finetuned_source_B.pt"), map_location="cpu"))
    netC.load_state_dict(torch.load(os.path.join(dir_name,"finetuned_source_C.pt"), map_location="cpu"))

    return nn.Sequential(base_model, netB, netC)  


def load_decision_office_home_model(args):
    #note : model structure for dine and decision is same. so we can use dine model
    
    if args.model_name[0:3] == 'res':
        base_model = dine_office_home_network.ResBase(res_name=args.model_name)
    elif args.model_name[0:3] == 'vgg':
        base_model = dine_office_home_network.VGGBase(vgg_name=args.model_name)

    netB = dine_office_home_network.feat_bootleneck(type=args.classifier, feature_dim=base_model.in_features, bottleneck_dim=args.bottleneck)
    netC = dine_office_home_network.feat_classifier(type=args.layer, class_num = args.num_classes, bottleneck_dim=args.bottleneck, p=args.droprate)
    
    dir_name = os.path.dirname(args.model_path)
    
    base_model.load_state_dict(torch.load(os.path.join(dir_name,"source_F.pt"), map_location="cpu"))
    netB.load_state_dict(torch.load(os.path.join(dir_name,"source_B.pt"), map_location="cpu"))
    netC.load_state_dict(torch.load(os.path.join(dir_name,"source_C.pt"), map_location="cpu"))

    return nn.Sequential(base_model, netB, netC)  



def get_model(model_name, droprate=0.005, channels=3, num_classes=10):

    if model_name == "resnet18":
        return ResNet18(p=droprate, channels=channels,num_classes=num_classes)
    elif model_name == "resnet18_source":
        return resnet18(p=droprate, channels=channels)
    elif model_name =="wideresnet":
        return  wrn.WideResNet(depth=28, num_classes= num_classes , widen_factor=2, dropRate=0.0, dad_droprate=droprate)
    #elif model_name =="wideresnetvar":
        #wide resnet var is used for stl dataset
    #    _net_builder = build_WideResNetVar(2,depth=28, widen_factor=2,leaky_slope=0.1, dropRate=0, use_embed=False, is_remix=False, dad_droprate=droprate)
    #    return  _net_builder.build(num_classes)
    elif model_name == "resnet34":
        return ResNet34(p=droprate, channels=channels, num_classes=num_classes)
    elif model_name == "resnet50":
        return rersnet50.resnet50(p=droprate)
    elif model_name == "vgg16":
        return VGG('VGG16')
    else:
        raise Exception ("model not defined" , model_name)

def get_norm_layer(args):
    mean, std = get_mean_and_std(args)
    norm_layer = Normalize(mean, std=std)
    return norm_layer

def get_normalized_model(args):
    
    norm_layer = get_norm_layer(args)
      
    if args.method == "shot_digit":
        base_model = load_shot_digit_model(args)
    elif args.method in [ "shot_office"]:
        base_model = load_shot_office_model(args)

    elif args.method in ["dine_office_home" , "dine_office"]:
        base_model = load_dine_office_home_model(args)
    elif args.method in ["decision_office_home", "decision_office"]:
        base_model = load_decision_office_home_model(args)

    elif args.method =="zskt":
        base_model = load_zskt_model(args)
    elif args.method=="vanila" and args.dataset in ["tiny_imagenet" , "cub"]:
        base_model = resnet_224.Resnet34(args.num_classes)
        state_dict = torch.load(args.model_path, map_location="cpu")
        base_model.load_state_dict(state_dict)
    elif args.method=="vanila" and args.dataset =="oxford_pet":
        base_model = cnn_model()
        state_dict = torch.load(args.model_path, map_location="cpu")
        base_model.load_state_dict(state_dict)
    elif args.method == "unsupervised" and args.dataset=="cifar10":
        from models.unsupervised.models import ClusteringModel
        from models.unsupervised.resnet_cifar import resnet18
        backbone = resnet18()
        base_model = ClusteringModel(backbone, 10, 1)
        state_dict=  torch.load(args.model_path, map_location="cpu")
        base_model.load_state_dict(state_dict)
    elif args.method == "unsupervised" and args.dataset=="stl":
        from models.unsupervised.models import ClusteringModel
        from models.unsupervised.resnet_stl import resnet18
        backbone = resnet18()
        base_model = ClusteringModel(backbone, 10, 1)
        state_dict=  torch.load(args.model_path, map_location="cpu")
        base_model.load_state_dict(state_dict)
    else: 
        if args.model_path is not None:
            base_model = get_model(args.model_name , droprate=args.droprate, channels=args.channels,num_classes=args.num_classes)
            if args.model_name in ["wideresnet", "wideresnetvar"]:
                checkpoint = torch.load(args.model_path , map_location="cpu")
                load_model = checkpoint['ema_model']

                new_state_dict = OrderedDict()
                for k, v in load_model.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v    

                base_model.load_state_dict(new_state_dict)

            else:
                state_dict = torch.load(args.model_path, map_location="cpu")
                base_model.load_state_dict(state_dict)
        
    
    return nn.Sequential(norm_layer, base_model)

def get_correct(outputs, labels):

    ## Correct Predicitons in a given batch
    _, pred = torch.max(outputs, 1)
    correct = (pred == labels).float().sum(0).item()
    return correct

    

def get_mean_and_std(args):
    if args.method in ["fixmatch" , "flexmatch"] and args.dataset=="cifar10":
        mean = [x/255. for x in [125.3, 123.0, 113.9]] 
        std  = [x/255. for x in [63.0, 62.1, 66.7]] 

    elif args.method in ["fixmatch" , "flexmatch"] and args.dataset=="svhn":
        #ref : https://github.com/TorchSSL/TorchSSL/blob/5cdf9bc76fbb859bc2521a7703a3f76ae2965119/datasets/ssl_dataset.py#L25
        mean=[0.4380, 0.4440, 0.4730]
        std=[0.1751, 0.1771, 0.1744]
    
    #TODO add mean and std for fixmatch and flexmatch
    
    elif args.method == "dafl" and ( args.dataset=="cifar100" or args.dataset=="cifar10") :
        mean =[0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.method == "dafl" and args.dataset=="svhn" :
        mean =[0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.method == "zskt" and ( args.dataset=="cifar100" or args.dataset=="cifar10") :
        mean =[0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.method == "zskt" and args.dataset=="svhn" :
        mean =[0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
    
    elif args.method == "shot_digit" :
        mean = [0.5]
        std = [0.5]
    elif args.method in ["shot_office" , "dine_office_home", "decision_office_home" , "dine_office" , "decision_office"]:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.method in ["vanila"] and args.dataset=="svhn":
        #mean = [0.5]
        #std = [0.5]
        mean , std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
    elif args.method in ["vanila"] and args.dataset=="cifar10":
        #mean = [0.4914, 0.4822, 0.4465]
        #std = [0.2023, 0.1994, 0.2010]
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

    elif args.method in ["vanila"] and args.dataset=="fmnist":
        #mean = [0.5]
        #std = [0.5]
        mean , std = (0.2860,), (0.3530,)
    elif args.method in ["vanila"] and args.dataset=="mnist":
        #mean = [0.5]
        #std = [0.5]
        mean , std = (0.1307,), (0.3081,)
    elif args.method == 'vanila' and args.dataset == "tiny_imagenet":
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]
    elif args.method == 'vanila' and args.dataset == "oxford_pet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.method =="vanila" and args.dataset=="cub":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.method=="unsupervised" and args.dataset=="cifar10":
        mean , std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    elif args.method=="unsupervised" and args.dataset=="stl":
        mean , std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        print("no mean defined for method  and dataset", args.method,dataset)
    return mean, std


#worker function used by dataloader to create fixed seed value
#ref : https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def fix_seed_value(seed):
    #fix seed value for reproducibility
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    
def get_adv_data(args, dataloader):
    #generate adversarial data based on different conditions
    if args.attack_mode == "only_detector":
        #if attacker only attacks detector
        return create_adv_data_detector(dataloader, args.detector ,args.dataset, args.attack)
    if args.attack_mode == "only_classifier":
        #if attacker only attacks detector
        return create_adv_data_classifier(dataloader ,args.dataset, args.attack, args.model)
    if args.attack_mode == "detector_classifier":
        #attack detector then classifer
        return create_adv_data_detector_classifier(dataloader,args.dataset, args.attack,args.detector , args.model , args.batch_size)
    if args.attack_mode == "classifier_detector":
        #attack classifer then detector
        return create_adv_data_classifier_detector(dataloader,args.dataset, args.attack,args.detector , args.model , args.batch_size)
    

def get_logger(file_path):

    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

def update_channels_and_num_classe_from_dataset(args):
    args.channels = 3
    args.num_classes =10
    if args.dataset in ["fmnist", "mnist"] :
        args.channels=1
    if args.dataset in ["cifar100" ]:
        args.num_classes = 100
    if args.method in ["shot_office" , "dine_office" , "decision_office"]:
        args.num_classes =31
    if args.dataset =="tiny_imagenet":
        args.num_classes =200
    if args.dataset =="oxford_pet":
        args.num_classes =37
    if args.dataset=="cub":
        args.num_classes =200
    if args.method in ["dine_office_home" , "decision_office_home"]:
        args.num_classes =65
    

    print("dataset channels :", args.channels)

    return args

if __name__ == '__main__':
    test = load_adv_data("data/dafl/cifar10/resnet18/PGD_data.pth")
    print(test[0])