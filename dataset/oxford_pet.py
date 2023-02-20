import torch
from glob import glob
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import os

def load_image(filename) :
    img = Image.open(filename)
    img = img.convert('RGB')
    return img


class Databasket():
    "Helper class to ensure equal distribution of classes in both train and validation datasets"
    
    def __init__(self,val_split=0.4):

        data = []
        labels = []
        filenames = glob('/media2/inder/dad_shubham/data-free-defense/clean_data/oxford_pet/images/*.jpg')
        
        
        classes = set()
        # Load the images and get the classnames from the image path
        for image in filenames:
            class_name = os.path.basename(image)
            i = class_name.rfind("_")
            class_name = image[0:i]
            classes.add(class_name)
            img = load_image(image)

            data.append(img)
            labels.append(class_name)

        # convert classnames to indices
        class2idx = {cl: idx for idx, cl in enumerate(classes)}        
        labels = torch.Tensor(list(map(lambda x: class2idx[x], labels))).long()

        data = list(zip(data, labels))
        num_cl =  len(classes)
    
        # Apply transformations to the train dataset
        train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # apply the same transformations to the validation set, with the exception of the
        # randomized transformation. We want the validation set to be consistent
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        class_values = [[] for x in range(num_cl)]
        
        # create arrays for each class type
        for d in data:
            class_values[d[1].item()].append(d)
            
        self.train_data = []
        self.val_data = []
        
        # put (1-val_split) of the images of each class into the train dataset
        # and val_split of the images into the validation dataset
        for class_dp in class_values:
            split_idx = int(len(class_dp)*(1-val_split))
            self.train_data += class_dp[:split_idx]
            self.val_data += class_dp[split_idx:]
            
        self.train_ds = PetDataset(self.train_data, transforms=train_transforms)
        self.val_ds = PetDataset(self.val_data, transforms=val_transforms)


class PetDataset(Dataset):
    "Dataset to serve individual images to our model"
    
    def __init__(self, data, transforms=None):
        self.data = data
        self.len = len(data)
        self.transforms = transforms
    
    def __getitem__(self, index):
        img, label = self.data[index]
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, label
    
    def __len__(self):
        return self.len



if __name__ == '__main__':
    dataset = Databasket()
    print(dataset.train_ds[23])