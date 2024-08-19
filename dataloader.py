"""Lightning dataloader for text and image data."""

import logging
import os
from typing import Any, Union

import joblib
import lightning as pl
import numpy as np
from lightning import seed_everything
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision
import torch

import torchvision
import ssl
import torch

ssl._create_default_https_context = ssl._create_stdlib_context
from torch.utils.data import Dataset, DataLoader
from config import HYPERPARAMETERS
from sklearn.preprocessing import LabelEncoder
pil_transform = transforms.Compose([transforms.PILToTensor()])

logger = logging.getLogger()


seed = seed_everything(21, workers=True)


class Encoder:
    def __init__(self):
        self.hyperparameters = HYPERPARAMETERS

    def load_labelencoder(self):
        """
        Function to load the label encoder from s3
        Returns:
        """
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        le = LabelEncoder()
        le.fit(classes)
        return le


class CifarDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset = None,  transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cifar = dataset
        self.transform = transform

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.cifar.data[idx]
        label = self.cifar.targets[idx]

        if self.transform:
            img = self.transform(img)

        sample = {'IMG': img, 'NID': idx, "GT": label}

        return sample



class ImgDataModule(pl.LightningDataModule):
    """Own DataModule form the pytorch lightning DataModule."""

    def __init__(self, hyperparameters: dict):
        """
        Init if the Data Module
        Args:
            data_path: dataframe with the data
            hyperparameters:  Hyperparameters
        """
        super().__init__()
        self.data_dir ="./data"
        self.hyperparameters = hyperparameters
        self.batch_size = hyperparameters["batch_size"]
        self.hyperparameters = hyperparameters
        self.num_classes = hyperparameters["num_classes"]
        # Augmentation policy for training set
        self.augmentation = transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                #transforms.Resize(size=224),
                transforms.RandomResizedCrop(size=300, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomInvert(),
                #transforms.RandomAdjustSharpness(0.05),
                #transforms.RandomAutocontrast(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5], [0.5]),
            ]
        )
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                transforms.Resize(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5], [0.5]),
            ]
        )

    def prepare_data(self):
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage in ['fit', "validate"] or stage is None:
            cifar_full = CifarDataset(torchvision.datasets.CIFAR10(self.data_dir, train=True),transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])
            self.dataset_train = self.cifar_train
            self.dataset_val = self.cifar_val

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar_test = torchvision.datasets.CIFAR10(self.data_dir, train=False)
            self.dataset_test = CifarDataset(self.cifar_test,transform=self.transform)


    def train_dataloader(self) -> DataLoader:
        """
        Define the training dataloader
        Returns:
            training dataloader
        """


        print(f"Length of the train dataset: {len(self.dataset_train)}")
        return torch.utils.data.DataLoader(self.dataset_train,
                                           batch_size=self.batch_size,
                                          num_workers=os.cpu_count(),
                                          shuffle=True, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        """
        Define the validation dataloader
        Returns:
            validation dataloader
        """


        print(f"Length of the val dataset: {len(self.dataset_val)}")
        return torch.utils.data.DataLoader(self.dataset_val,
                                           batch_size=self.batch_size,
                                          num_workers=os.cpu_count(),
                                          shuffle=False, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        """
        Define the test dataloader
        Returns:
            test dataloader
        """

        print(f"Length of the test dataset: {len(self.dataset_test)}")
        return torch.utils.data.DataLoader(self.dataset_test,
                                           batch_size=self.batch_size,
                                          num_workers=os.cpu_count(),
                                          shuffle=False,persistent_workers=True)