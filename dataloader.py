"""Lightning dataloader for text and image data."""

import logging
import os

import lightning as pl
from lightning import seed_everything
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import ssl
import torch

ssl._create_default_https_context = ssl._create_stdlib_context
from torch.utils.data import Dataset, DataLoader
from config import HYPERPARAMETERS
from sklearn.preprocessing import LabelEncoder
pil_transform = transforms.Compose([transforms.PILToTensor()])

logger = logging.getLogger()


seed = seed_everything(21, workers=True)

import torch
from torch.utils.data import DataLoader

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path, _ = self.samples[index]  # Get image path
        filename = os.path.basename(path)  # Extract filename
        return img, label, filename  # Return filename along with data

# Wrapper class to apply transforms
class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data, label = self.subset[idx]  # Access the data and label
        if self.transform:
            data = self.transform(data)
        return data, label

def calculate_mean_std(dataset, batch_size=1536):
    """
    Calculate the mean and standard deviation of a PyTorch dataset.

    Args:
        dataset (Dataset): PyTorch dataset (e.g., training dataset).
        batch_size (int): Batch size for the DataLoader.

    Returns:
        tuple: Mean and standard deviation tensors.
    """
    # Use DataLoader to iterate through the dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0

    for data, _ in loader:  # Assuming dataset returns (data, labels)
        batch_samples = data.size(0)  # Number of samples in the batch
        data = data.view(batch_samples, data.size(1), -1)  # Flatten H x W to single dimension

        mean += data.mean(dim=(0, 2)) * batch_samples  # Weighted sum
        std += data.std(dim=(0, 2)) * batch_samples  # Weighted sum
        n_samples += batch_samples

        print("# of samples processed: ", n_samples)

    mean /= n_samples  # Compute final mean
    std /= n_samples  # Compute final std

    return mean.numpy(), std.numpy()



class Encoder:
    def __init__(self, HYPERPARAMETERS=HYPERPARAMETERS):
        self.hyperparameters = HYPERPARAMETERS
        self.data_dir = self.hyperparameters["data_dir"]

    def load_labelencoder(self):
        """
        Function to load the label encoder from s3
        Returns:
        """

        classes = datasets.ImageFolder(root=self.data_dir + "/train", transform=None).classes
        #classes = classes.tolist()
        #classes.append("XXX_Unknown_Label") # ToDo Think about

        le = LabelEncoder()
        le.fit(classes)
        return le


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
        self.hyperparameters = hyperparameters
        self.data_dir = hyperparameters["data_dir"]
        self.predict_data_dir = hyperparameters["predict_data_dir"]
        self.batch_size = hyperparameters["batch_size"]
        self.hyperparameters = hyperparameters
        self.num_classes = hyperparameters["num_classes"]
        # Augmentation policy for training set
        self.augmentation = transforms.Compose(
            [
                #torchvision.transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),  # Convert 1 channel (BW) to 3 channels (RGB)
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(size=448, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomInvert(),
                transforms.RandomAdjustSharpness(0.5),
                transforms.RandomAutocontrast(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                #transforms.Normalize([4.852536949329078e-05], [9.840591519605368e-05]),#calc for this train dataset
            ]
        )
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose(
            [
                #torchvision.transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),  # Convert 1 channel (BW) to 3 channels (RGB)
                transforms.Resize((448, 448)),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                #transforms.Normalize([4.852536949329078e-05], [9.840591519605368e-05]),
            ]
        )



    def prepare_data(self):
        self.zoo_train = ImageFolderWithPaths(root=self.data_dir + "/train", transform=self.augmentation)
        self.zoo_test = ImageFolderWithPaths(root=self.data_dir + "/test", transform=self.transform)
        self.zoo_val = ImageFolderWithPaths(root=self.data_dir + "/val", transform=self.transform)
        self.zoo_predict =ImageFolderWithPaths(root=self.predict_data_dir + "/predict", transform=self.transform)





    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage in ['fit', "validate"] or stage is None:
            self.dataset_train = self.zoo_train
            self.dataset_val = self.zoo_val


        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.dataset_test = self.zoo_test

        if stage == 'predict' or stage is None:
            self.dataset_predict = self.zoo_predict


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
    def predict_dataloader(self) -> DataLoader:
        """
        Define the test dataloader
        Returns:
            test dataloader
        """

        print(f"Length of the predict dataset: {len(self.dataset_predict)}")
        return torch.utils.data.DataLoader(self.dataset_predict,
                                           batch_size=self.batch_size,
                                          num_workers=os.cpu_count(),
                                          shuffle=False,persistent_workers=True)

if __name__ == "__main__":
    data_dir = "experiments/dataset012"
    transform = transforms.Compose(
        [
            # torchvision.transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),  # Convert 1 channel (BW) to 3 channels (RGB)
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset_train = datasets.ImageFolder(root=data_dir+"/train", transform=transform)
    dataset_test = datasets.ImageFolder(root=data_dir+"/test", transform=transform)
    dataset_val = datasets.ImageFolder(root=data_dir+"/val", transform=transform)

    print(dataset_train.classes)
    print(dataset_test)
    print(dataset_val)
    image_paths = [sample[0] for sample in dataset_train.samples]
    print(image_paths)
    #zoo_train, zoo_val, zoo_test = random_split(dataset, [0.7, 0.15, 0.15])
    #mean, std = calculate_mean_std(dataset_train)

    #print("#####" * 10)
    #print(f"Mean: {mean}, Std: {std}")
    #print("#####" * 10)