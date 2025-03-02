import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from matplotlib import pyplot as plt
import random
from tqdm import tqdm


from mask_generate_anomaly import generate_anomaly_mask

from generate_anomaly import fpi_mask_generate_anomaly

class anomaly_transform(object):
    def __init__(self, mypseudo=True, anomaly_source_images=None):
        self.anomaly_weight = [-0.999, -0.99, 2, 3]

        self.anomaly_source_images = anomaly_source_images
        self.mypseudo = mypseudo

    def __call__(self, image, mask):
        image = np.array(image)
        if self.mypseudo:
            random_choice = random.choice(self.anomaly_weight)

            new_img = generate_anomaly_mask(image, random_choice, mask)
            new_img = Image.fromarray(new_img)

            return new_img, torch.Tensor([0,1])
        else:
            
            anomaly_source_image = random.choice(self.anomaly_source_images)
            new_img = fpi_mask_generate_anomaly(image, anomaly_source_image, mask)
            new_img = Image.fromarray(new_img)

            return new_img, torch.Tensor([0,1])
        

class ChestXrayDataSet(torch.utils.data.Dataset):
    def __init__(self,  
                dataframe,
                img_size=(224, 224), 
                mode='train', 
                ):
        """
        dataframe: pandas dataframe
        img_size: tuple
        model: ["train", "val", "test"]
        """

        self.data = []
        self.img_size = img_size  
        self.transforms = self.get_data_transforms(img_size, mode)

        self.dataframe = dataframe
        self.labels = self.dataframe.iloc[:, 2:-1].values  # Assuming columns 2 to n-1 are labels

        self.load_data()

        self.anomaly_transform = anomaly_transform()

        mask_1 = torch.zeros((224, 224, 1))
        mask_1[:, 0:112, :] = 1

        mask_2 = torch.zeros((224, 224, 1))
        mask_2[:, 112:, :] = 1

        mask_3 = torch.zeros((224, 224, 1))
        mask_3[0:120, :, :] = 1

        mask_4 = torch.zeros((224, 224, 1))
        mask_4[120:, :, :] = 1

        mask_5 = torch.zeros((224, 224, 1))
        mask_5[:, :, :] = 1

        self.masks = [mask_1, mask_2, mask_3, mask_4, mask_5]
        self.position_names = ["left lung", "right lung", "upper lung", "lower lung", ""]


    def load_data(self):

        for idx in tqdm(range(len(self.dataframe))):
            img_path = self.dataframe.iloc[idx]['FilePath'].replace('..', '/kaggle')
            img = Image.open(img_path).convert('L').resize(self.img_size)
            img = np.array(img)
            img = np.stack((img,)*3, axis=-1)
            img = Image.fromarray(img)
        
            label = torch.FloatTensor(self.labels[idx])
        
            self.data.append((img, label))
        print('%d data loaded from: %s' % (len(self.data), self.dataframe))

    # Function to get data transforms
    def get_data_transforms(self, img_size, mode):
        if mode == "train":
            train_transform = transforms.Compose([
                transforms.Resize((img_size[0], img_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return train_transform
        else:
            val_test_transform = transforms.Compose([
                transforms.Resize((img_size[0], img_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return val_test_transform
        
    def __getitem__(self, index):
        img, label = self.data[index]
        
        position_name = self.position_names
        mask = self.masks
        masks = []
        for m in mask:
            m = m.permute(2,0,1)
            masks.append(m)
        img= self.transforms(img)

        return img,  label.long(), masks, position_name

    def __len__(self):
        return len(self.data)


