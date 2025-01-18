"""Train, validate and test loop using the lightning framework."""

import logging
import os
from typing import Any, Dict, Sequence, Union

import lightning as pl
import pandas as pd
import torch
from dataloader import Encoder
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from model_arc import ResNet18
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from transformers import AdamW

logger = logging.getLogger()

seed = seed_everything(21, workers=True)


class SklearnMetricsCallback(pl.Callback):
    def __init__(self, label_encoder, hyperparameters):
        super().__init__()
        self.label_encoder = label_encoder
        self.hyperparameters = hyperparameters

    def on_test_epoch_end(self, trainer, pl_module):
        output_list = []
        ground_truths_list = []
        for batch_item in pl_module.pred_list:
            output_list.append(batch_item["outputs"])
            ground_truths_list.append(batch_item["ground_truth"])
        output_tensor = torch.concat(output_list, dim=0)
        ground_truths_tensor = torch.concat(ground_truths_list, dim=0)
        self._sklearn_metrics(output_tensor, ground_truths_tensor, "test")

    def _sklearn_metrics(
        self, output: torch.Tensor, ground_truths: torch.Tensor, mode: str
    ):
        logger.info(("output shape", output.shape))
        logger.info(("ground_truths shape", ground_truths.shape))
        model_dir = self.hyperparameters["model_dir"]

        softmax = nn.Softmax(dim=1)
        preds = softmax(output).argmax(dim=1)
        confis = softmax(output).detach().cpu().numpy()
        y_pred = self.label_encoder.inverse_transform(preds.detach().cpu().numpy())
        y_true = self.label_encoder.inverse_transform(ground_truths.detach().cpu().numpy())

        report = classification_report(y_true, y_pred, output_dict=True)
        report_confusion_matrix = confusion_matrix(y_true, y_pred, labels=list(self.label_encoder.classes_))

        self.save_reports(model_dir, mode, report_confusion_matrix, report)
        self.save_test_evaluations(model_dir, mode, y_pred, y_true, confis)

    def save_reports(self, model_dir, mode, report_confusion_matrix, report):
        """Save classification report and confusion matrix to csv file.

        Args:
            model_dir: path
            mode: train, test or val
            report_confusion_matrix: sklearn confusion matrix
            report: sklear classification report
        Returns:

        """
        df_cm = pd.DataFrame(report_confusion_matrix)
        df_cr = pd.DataFrame(report).transpose()
        df_cm.to_csv(f"{model_dir}/{mode}_confusion_matrix.csv", sep=";")
        df_cr.to_csv(f"{model_dir}/{mode}_classification_report.csv", sep=";")
        logger.info("Confusion Matrix and Classication report are saved.")

    def save_test_evaluations(self, model_dir, mode, y_pred, y_true, confis):
        """
        Save a pandas dataframe with prediction and ground truth of the test dataset
        Args:
            model_dir:
            mode:
            y_pred:
            y_true:
            confis:
        Returns:
        """
        df_test = pd.DataFrame()
        df_test["pred"] = y_pred
        df_test["confidence"] = confis.max(axis=1)
        df_test["label"] = y_true
        df_test.to_csv(f"{model_dir}/{mode}_labels_predictions.csv", sep=";")
        logger.info("The label predictions are saved.")


class LitModel(pl.LightningModule):
    """Lightning model for classification."""

    def __init__(
        self,
        hyperparameters: dict,
    ):
        super().__init__()
        self.hyperparameters = hyperparameters
        EC = Encoder()
        self.label_encoder = EC.load_labelencoder()
        self.learning_rate = self.hyperparameters["learning_rate"]
        self.batch_size = self.hyperparameters["batch_size"]
        self.num_classes = len(self.label_encoder.classes_)
        self.hyperparameters["num_classes"] = self.num_classes
        self.module = ResNet18(
             hyperparameters=self.hyperparameters
        )
        # Classification
        self.criterion = nn.CrossEntropyLoss()
        metrics = MetricCollection(
            [
                MulticlassAccuracy(self.num_classes),
                MulticlassPrecision(self.num_classes),
                MulticlassRecall(self.num_classes),
                MulticlassF1Score(self.num_classes),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.pred_list = []
        self.save_hyperparameters()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward path, calculate the computational graph in the forward direction.

        Used for train, test and val.
        Args:
            y: tensor with text data as tokens
        Returns:
            computional graph

        """
        return self.module(x)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict:
        """
        Call the eval share for training
        Args:
            batch: tensor
        Returns:
            dict with loss, outputs and ground_truth
        """
        return self._shared_eval_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        """
        Call the eval share for validation
        Args:
            batch:
            batch_idx:
        Returns:
            dict with loss, outputs and ground_truth
        """
        return self._shared_eval_step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        """
        Call the eval share for test
        Args:
            batch:
            batch_idx:
        Returns:
            dict with loss, outputs and ground_truth
        """
        ret = self._shared_eval_step(batch, "test")
        self.pred_list.append(ret)
        return ret

    def _shared_eval_step(self, batch: Dict[str, torch.Tensor], mode: str) -> Dict:
        """Calculate the desired metrics.

        Args:
            batch: tensor
            mode: train, test or val
        Returns:
            dict with loss, outputs and ground_truth

        """
        img , ground_truth = batch
        out = self.forward(img)
        if mode == "train":
            loss = self.criterion(out, ground_truth)
            output = self.train_metrics(out, ground_truth)
            self.log_dict(output)
            self.train_metrics.update(out, ground_truth)
        elif mode == "val":
            loss = self.criterion(out, ground_truth)
            output = self.val_metrics(out, ground_truth)
            self.log_dict(output)
            self.val_metrics.update(out, ground_truth)
        elif mode == "test":
            loss = self.criterion(out, ground_truth)
            output = self.test_metrics(out, ground_truth)
            self.log_dict(output)
            self.test_metrics.update(out, ground_truth)
            # reset predict list
            # self.pred_list = []

        return {"outputs": out, "loss": loss, "ground_truth": ground_truth}

    def _epoch_end(self, mode: str):
        """
        Calculate loss and metricies at end of epoch
        Args:
            mode:
        Returns:
            None
        """
        if mode == "val":
            output = self.val_metrics.compute()
            self.log_dict(output)
            self.val_metrics.reset()
        if mode == "train":
            output = self.train_metrics.compute()
            self.log_dict(output)
            self.train_metrics.reset()
        if mode == "test":
            output = self.test_metrics.compute()
            self.log_dict(output)
            self.test_metrics.reset()


    def on_test_epoch_end(self) -> None:
        """
        Calculate the metrics at the end of epoch for test step
        Args:
            outputs:
        Returns:
            None
        """
        self._epoch_end("test")

    def on_validation_epoch_end(self):
        """
        Calculate the metrics at the end of epoch for val step
        Args:
            outputs:
        Returns:
            None
        """
        self._epoch_end("val")

    def on_train_epoch_end(self):
        """
        Calculate the metrics at the end of epoch for train step
        Args:
            outputs:
        Returns:
            None
        """
        self._epoch_end("train")

    def configure_optimizers(self) -> Any:
        """
        Configure the optimizer
        Returns:
            optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.hyperparameters["weight_decay"])
        scheduler = StepLR(optimizer, step_size=self.hyperparameters["step_size"], gamma=self.hyperparameters["gamma"])
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def configure_callbacks(self) -> Union[Sequence[pl.pytorch.Callback], pl.pytorch.Callback]:
        """Configure Early stopping or Model Checkpointing.

        Returns:

        """
        early_stop = EarlyStopping(
            monitor="val_MulticlassAccuracy", patience=self.hyperparameters["patience"], mode="max"
        )
        checkpoint = ModelCheckpoint(
            monitor="val_MulticlassAccuracy",
            mode="max",
            dirpath=self.hyperparameters["model_dir"] + "/",
            filename=self.hyperparameters["model_filename"],
        )

        sklearn = SklearnMetricsCallback(label_encoder=self.label_encoder, hyperparameters=self.hyperparameters)
        return [early_stop, checkpoint, sklearn]