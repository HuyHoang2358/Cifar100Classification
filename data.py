import torch
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms

class CIFAR100Dataset():
    def __init__(self,batch_size = 256):
        self.transform = transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))
                        ])
        self.batch_size = batch_size
        self.train_set = datasets.CIFAR100("/",train=True, download=True, transform = self.transform)
        self.val_set = datasets.CIFAR100("/",train=False, download=True, transform = self.transform)

    def data_Loader(self,data_set,shuffle = True):
        return torch.utils.data.DataLoader(data_set, batch_size= self.batch_size, shuffle=shuffle)

    def get_train_loader(self):
        return self.data_Loader(self.train_set)

    def get_val_loader(self):
        return self.data_Loader(self.val_set)