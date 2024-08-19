"""Config file with the Hyperparameter dict."""

import logging

logger = logging.getLogger(__name__)

OUT_URI = "output/"


HYPERPARAMETERS = {
    "freeze_backbones": "freeze", # "freeze"
    "num_freeze_layers_resnet": 100,
    "devices": -1,
    "profiler": "advanced",
    "test_mode": "on",
    "val_mode": "on",
    "limit_batches": None,
    "precision": "16-mixed",
    "out_uri": OUT_URI,
    "batch_size": 1536,
    "max_epochs": 100,
    "text_max_length": 512,
    "model_dir": "./artefacts",
    "learning_rate": 1e-3,
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "resnet18_name": "resnet18.pth",
    "load_model": True,
    "model_path": " ",
    "patience": 2,
    "num_workers": 8,
    "num_classes": 10,
    "model_filename": "best",
    "img_size": 224,
    "dropout": 0.25,
    "continue": 0,
    "endpoint_mode": False,
    "selected_model": "resnet",
    "input_dir": "lightning_data/",
}