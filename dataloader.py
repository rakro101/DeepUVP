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


class Encoder:
    def __init__(self):
        self.hyperparameters = HYPERPARAMETERS

    def load_labelencoder(self):
        """
        Function to load the label encoder from s3
        Returns:
        """
        classes = ['Acartiidae', 'Actiniaria', 'Actinopterygii', 'Aglaura', 'Amphipoda', 'Annelida', 'Atlanta', 'Bassia', 'Bivalvia', 'Calanidae', 'Calanoida', 'Calocalanus pavo', 'Candaciidae', 'Cavolinia inflexa', 'Centropagidae', 'Chaetognatha', 'Copilia', 'Corycaeidae', 'Coscinodiscus', 'Creseidae', 'Creseidae acicula', 'Ctenophora', 'Cumacea', 'Cymbulia peroni', 'Doliolida', 'Eucalanidae', 'Euchaetidae', 'Eumalacostraca', 'Evadne', 'Foraminifera', 'Fritillariidae', 'Gymnosomata', 'Haloptilus', 'Harosa', 'Harpacticoida', 'Heterorhabdidae', 'Hydrozoa', 'Hyperiidea', 'Insecta', 'Isopoda', 'Limacinidae', 'Liriope', 'Metridinidae', 'Mysida', 'Neoceratium', 'Noctiluca', 'Obelia', 'Oikopleuridae', 'Oithonidae', 'Oncaeidae', 'Ostracoda', 'Penilia', 'Phaeodaria', 'Physonectae', 'Podon', 'Pontellidae', 'Pyrosomatida', 'Rhincalanidae', 'Rhopalonema velatum', 'Salpida', 'Sapphirinidae', 'Solmundella bitentaculata', 'Temoridae', 'Tomopteridae', 'actinula', 'artefact', 'badfocus', 'bract_Abylopsis tetragona', 'bract_Diphyidae', 'bubble', 'calyptopsis', 'cirrus', 'cyphonaute', 'cypris', 'detritus', 'egg_Actinopterygii', 'egg_Mollusca', 'endostyle', 'ephyra', 'eudoxie_Abylopsis tetragona', 'eudoxie_Diphyidae', 'fiber', 'gonophore_Abylopsis tetragona', 'gonophore_Diphyidae', 'head_Chaetognatha', 'juvenile_Salpida', 'larvae_Annelida', 'larvae_Echinodermata', 'larvae_Luciferidae', 'larvae_Mysida', 'larvae_Porcellanidae', 'larvae_Stomatopoda', 'megalopa', 'metanauplii_Crustacea', 'nauplii_Cirripedia', 'nauplii_Crustacea', 'nectophore_Abylopsis tetragona', 'nectophore_Diphyidae', 'nectophore_Hippopodiidae', 'nectophore_Physonectae', 'nucleus', 'other_egg', 'other_living', 'part_Annelida', 'part_Cnidaria', 'part_Crustacea', 'part_Mollusca', 'part_Siphonophorae', 'pluteus_Echinoidea', 'pluteus_Ophiuroidea', 'protozoea_Eumalacostraca', 'protozoea_Penaeidae', 'protozoea_Sergestidae', 'seaweed', 'siphonula', 'tail_Appendicularia', 'tail_Chaetognatha', 'trunk_Appendicularia', 'zoea_Brachyura', 'zoea_Galatheidae']

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
        self.data_dir ="ZooScanNet/imgs"
        self.hyperparameters = hyperparameters
        self.batch_size = hyperparameters["batch_size"]
        self.hyperparameters = hyperparameters
        self.num_classes = hyperparameters["num_classes"]
        # Augmentation policy for training set
        self.augmentation = transforms.Compose(
            [
                #torchvision.transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),  # Convert 1 channel (BW) to 3 channels (RGB)
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(size=300, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomInvert(),
                #transforms.RandomAdjustSharpness(0.05),
                #transforms.RandomAutocontrast(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5], [0.5]),#calc for this train dataset
            ]
        )
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose(
            [
                #torchvision.transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),  # Convert 1 channel (BW) to 3 channels (RGB)
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5], [0.5]),
            ]
        )

    def prepare_data(self):
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        self.zoo_train, self.zoo_val, self.zoo_test = random_split(self.dataset, [0.7, 0.15, 0.15])

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage in ['fit', "validate"] or stage is None:
            self.dataset_train = self.zoo_train
            self.dataset_val = self.zoo_val

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.dataset_test = self.zoo_test


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