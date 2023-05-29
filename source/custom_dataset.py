import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch
import os
import torchvision
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class TarotDataset(Dataset):
    """Tarot cards dataset."""

    def __init__(self, csv_file, root_dir, transform=torchvision.transforms.ToTensor()):

        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        print("Dataset created")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data.iloc[idx, 0])
        
        img  = Image.open(img_name)
        print(type(img))

        label = self.data.iloc[idx, 1]
        image = self.transform(img=img)
        print(type(image))

        sample = {'image': self.transform(img), 'label': label}

        return sample

    def display(self, idx):

        img_name = os.path.join(self.root_dir,
                                self.data.iloc[idx, 0])
        
        img = mpimg.imread(img_name)
        label = self.data.iloc[idx, 1]
        plt.imshow(img)
        plt.title(label)
        plt.show()