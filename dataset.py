import glob
import os
import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np

# the image and mask path
train_img_path = "./data/train/images/"
train_mask_path = "./data/train/masks/"

test_img_path = "./data/test/images/"
test_mask_path = "./data/test/masks/"

Transform_train = transforms.Compose([  # transforms.Resize((256, 256)),
    # transforms.RandomRotation((-15, 15)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomAffine(degrees=(-15, 15), translate=(0.2, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor()
])

Transform_test = transforms.Compose([  # transforms.Resize((256, 256)),
    transforms.ToTensor()
])


class LonglegsDataset(Dataset):
    def __init__(self, split='train'):
        assert split in ['train', 'test']
        self.split = split

        if self.split == 'test':
            self.img_path = test_img_path
            self.mask_path = test_mask_path
            self.transform = Transform_test
            self.file_names = os.listdir(test_img_path)
            # get only image names
            self.img_names = []
            for name in self.file_names:
                if name[-4:] == ".jpg":
                    self.img_names.append(name)
            self.mask_names = os.listdir(test_mask_path)
        else:
            self.img_path = train_img_path
            self.mask_path = train_mask_path
            self.transform = Transform_train
            # contain all images and json files' name
            self.file_names = os.listdir(train_img_path)
            # get only image names
            self.img_names = []
            for name in self.file_names:
                if name[-4:] == ".jpg":
                    self.img_names.append(name)
            self.mask_names = os.listdir(train_mask_path)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_dir = os.path.join(self.img_path, self.img_names[idx])
        mask_dir = os.path.join(self.mask_path, self.mask_names[idx])

        img = Image.open(img_dir).convert("L")
        mask = Image.open(mask_dir)
        img = transforms.ColorJitter(brightness=0.3, contrast=0.3)(img)
        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask
