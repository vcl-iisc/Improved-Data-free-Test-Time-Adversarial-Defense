
import wandb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
#import svhn datasets from torchvision.datasets
from torchvision.datasets import SVHN , CIFAR10 , FashionMNIST, MNIST
import argparse
from dataset.cub200 import Cub2011
from models.resnet import ResNet18

import utils

def train_one_epoch(model, optimizer,scheduler, data_loader, device):
    model.train()
    train_loss = 0
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
    return train_loss / len(data_loader)


def validate(model, data_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)

    # log validation loss and accuracy on wandb
    
    wandb.log({'val_loss': val_loss / len(data_loader), 'val_acc': correct / len(data_loader.dataset)})

    return val_loss / len(data_loader), correct / len(data_loader.dataset)


# define test function   
def test(model, test_loader):
    model.eval()
    correct = 0
    # get model device
    device = next(model.parameters()).device

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)

    print('Test accuracy: ', correct / len(test_loader.dataset))

    return correct / len(test_loader.dataset)

def get_mean_and_std(dataset):
    if dataset == "mnist":
        return (0.1307,), (0.3081,)
    elif dataset == "fmnist":
        return (0.2860,), (0.3530,)
    elif dataset == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    elif dataset == "svhn":
        return (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
    elif dataset == "stl":
        return (0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712)
    elif dataset =='cub':
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        raise ValueError("Invalid dataset name")

def create_model(model_name, channels):
     if model_name == "resnet18":
        return ResNet18(p=0, channels=channels,num_classes=10)    

import torch
from torchvision import transforms

data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    #random resize crop for fmnist datasets
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=None, shear=None, resample=False, fillcolor=0),
    transforms.ToTensor()
])


def get_data_loader(dataset , batch_size, image_size=224):
    mean, std = get_mean_and_std(dataset)
    train_transform = transforms.Compose([
        transforms.Resize(32),
        data_augmentation,
        transforms.Normalize(mean, std),
      ])
    
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)

    ])

    #load cifar10 train dataset if dataset is cifar10 else load svhn train dataset
    if dataset == 'cifar10':
        #load cifar10 train dataset
        train_dataset = CIFAR10(root='clean_data/cifar10', train=True, transform=train_transform, download=True)
        test_dataset = CIFAR10(root='clean_data/cifar10', train=False, transform=test_transform, download=True)
    elif dataset == 'svhn':
        #load svhn train dataset
        train_dataset = SVHN(root='clean_data/svhn', split='train', transform=train_transform, download=True)
        test_dataset = SVHN(root='clean_data/svhn', split='test', transform=test_transform, download=True)
    elif dataset == 'fmnist':
        #load fmnist train dataset
        train_dataset = FashionMNIST(root='clean_data/fmnist', train=True, transform=train_transform, download=True)
        test_dataset = FashionMNIST(root='clean_data/fmnist', train=False, transform=test_transform, download=True)
    elif dataset == 'mnist':
        #load mnist train dataset
        train_dataset = MNIST(root='clean_data/mnist', train=True, transform=train_transform, download=True)
        test_dataset = MNIST(root='clean_data/mnist', train=False, transform=test_transform, download=True)
    elif dataset =="cub":
        train_dataset = Cub2011(root  = "/media2/inder/dad_shubham/data-free-defense/clean_data/cub",train=True, transform=train_transform, download=True)
        test_dataset = Cub2011(root="/media2/inder/dad_shubham/data-free-defense/clean_data/cub",train=False, transform=test_transform, download=True)
    else:
        print('Dataset not supported')
        exit()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test_loader


# add python arguments use argparse, add arguments for batch size, learning rate, image size, epochs
def main(args):
    # set wandb config from args config
    wandb.init(project='svhn')
    config = wandb.config
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.image_size = args.image_size
    config.epochs = args.epochs

    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = get_data_loader(args.dataset, config.batch_size, config.image_size)

    if args.dataset == "fmnist" or args.dataset == "mnist":
        channels = 1
    else:
        channels = 3

    model = create_model(args.model_name, channels)
    model = model.to(config.device)
    model = nn.DataParallel(model)
    
    optimizer = torch.optim.SGD(model.module.parameters(), lr=config.lr, weight_decay=5e-4, momentum=0.9)
 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc = 0
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, optimizer, scheduler, train_loader, config.device)
        val_loss, val_acc = validate(model, val_loader, config.device)
        wandb.log({'train_loss': train_loss})
        
        #wandb log val loss and acc loss
        wandb.log({'val_loss': val_loss, 'val_acc': val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            # save best model as best_datasetname.pth
            torch.save(model.module.state_dict(), args.save_path)
        
        #early stop if val_acc is lesser than best_acc for 10 epochs
        if val_acc < best_acc:
            early_stop += 1
            if early_stop == 30:
                break
        else:
            early_stop = 0


    #test model
    test(model, test_loader)

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    #dataset name arg
    parser.add_argument('--dataset', type=str, default='cub')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    #boolean argument to decide whether to use wandb or not
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_path', type=str, default= None)
    parser.add_argument('--model_name', type=str, default= "resnet18")
    args = parser.parse_args()
    main(args)

    utils.fix_seed_value(args.seed)
    #command to run this Program
    #python train.py --dataset cifar10 --batch_size 32 --lr 0.001 --image_size 224 --epochs 100 --wandb True
