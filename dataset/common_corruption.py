#create custom dataset classs
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class CommonCorruption(Dataset):
    def __init__(self, root, corruption_name):
        self.root = root
        self.corruption_name = corruption_name
        self.images = np.load(f"{root}/{corruption_name}.npy")
        self.labels = np.load(f"{root}/labels.npy")
        
        self.labels = torch.from_numpy(self.labels)
        self.labels = self.labels.long()

        self.transforms = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.transforms(self.images[idx]), self.labels[idx]
    
    def get_corruption_name(self):
        return self.corruption_name
    
if __name__ == "__main__":
    dataset = CommonCorruption("clean_data/common_corruption/CIFAR-10-C", "gaussian_blur")
    print(dataset[0][0].shape)
    print(dataset[0][1])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    print(len(dataloader))
    for i, (images, labels) in enumerate(dataloader):
        print(images.shape)
        print(labels.shape)
        break
    



