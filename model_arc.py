"""Architecure Resnet lightning."""

import logging

import torch
from lightning import seed_everything
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

logger = logging.getLogger(__name__)

torch.manual_seed(21)
seed = seed_everything(21, workers=True)


class ResNet18(nn.Module):
    """ResNet 18 cut of the last layer for latent space representation."""

    def __init__(self, hyperparameters: dict):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights)
        logger.info(hyperparameters)
        logger.info(hyperparameters["num_classes"])
        self.hyperparameters = hyperparameters
        self.num_classes = hyperparameters["num_classes"]
        if (hyperparameters["freeze_backbones"] == "freeze"):
            logger.info("resnet layer weights are frozen")
            num_of_layers = len([par for par in self.model.parameters()])
            num_freeze = min(num_of_layers, hyperparameters["num_freeze_layers_resnet"])
            logger.info("first %s layers are frozen", num_freeze)
            param_counter = num_of_layers
            for param in self.model.parameters():
                if param_counter <= num_freeze:
                    param.requires_grad = False
                    print("param %s is frozen", param.shape)
                param_counter -= 1
        self.model.fc = nn.Identity()
        self.classifier = nn.Linear(512, self.num_classes)

    def forward(self, x):
        """Forward step for resnet18."""
        return self.classifier(self.model(x))
