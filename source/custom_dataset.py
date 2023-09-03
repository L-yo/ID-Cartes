import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datapoints as dp
from torchvision.utils import draw_bounding_boxes
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class TarotDataset(Dataset):

    def __init__(self, root_dir, csv_file, label_dict, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.labeled_cards = pd.read_csv(root_dir + csv_file)
        self.transform = transform
        self.label_dict = pd.read_csv(root_dir + label_dict)
        self.label_dict.set_index('label')

    def __len__(self):
        return self.labeled_cards.nunique()[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, "images/"+str(idx)+".png")
        image = io.imread(img_name)

        rows_of_interest = self.labeled_cards.loc[self.labeled_cards['image'] == str(idx)+".png"]

        labels = []
        boxes = []

        for line in rows_of_interest.to_numpy():
            labels.append(int(np.where(self.label_dict["label"] == line[1])[0][0]))
            boxes.append(line[2:])

        bboxes = dp.BoundingBox(np.asarray(boxes, dtype=int),
                                format="XYXY", 
                                spatial_size = torch.Size([np.size(image, 0), np.size(image, 1)]))


        target = {'boxes': bboxes, 'labels': labels, 'image_id': idx}

        sample =  [image, target]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def show_image(self, idx):
        sample = self.__getitem__(idx)
        # print(sample)
        image = transforms.functional.to_pil_image(sample[0])
        img = sample[0][:3]
        labs = []
        labels = sample[1]['labels'].tolist()
        print(self.label_dict)
        for lab in labels:
            labs.append(self.label_dict._get_value(lab, 'label'))
        bbox= sample[1]['boxes']
        img = draw_bounding_boxes(img, bbox, labs, width=3, colors=(255,255,0))
        image = transforms.ToPILImage()(img)
        image.show()


class ToTensor(object):

    def __call__(self, sample):
        image, target = sample[0], sample[1]
        labels = torch.as_tensor(target["labels"], dtype=torch.uint8)
        bboxes = torch.tensor(target["boxes"])
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return [torch.from_numpy(image), {"labels" : labels, "boxes" : bboxes}]
