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
from model_arc import ResNet18, ResNet34, ResNet50, EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from transformers import AdamW
from PIL import Image
from matplotlib import cm
import numpy as np

logger = logging.getLogger()

seed = seed_everything(21, workers=True)
import wandb
from lightning.pytorch.callbacks import Callback


class SklearnMetricsCallback(pl.Callback):
    def __init__(self, label_encoder, hyperparameters, logger):
        super().__init__()
        self.label_encoder = label_encoder
        self.hyperparameters = hyperparameters
        self.logger = logger

    def on_test_epoch_end(self, trainer, pl_module):
        output_list = []
        ground_truths_list = []
        image_list = []
        image_name_list = []
        for batch_item in pl_module.pred_list:
            output_list.append(batch_item["outputs"])
            ground_truths_list.append(batch_item["ground_truth"])
            image_list.append(batch_item["images"])
            image_name_list.append(batch_item["img_name"])


        image_tensor = torch.concat(image_list, dim=0)
        output_tensor = torch.concat(output_list, dim=0)
        ground_truths_tensor = torch.concat(ground_truths_list, dim=0)
        image_name_list = [item for entry in image_name_list for item in entry]
        self._sklearn_metrics(output_tensor, ground_truths_tensor, "test", image_name_list)
        self._log_images(output_tensor , ground_truths_tensor, image_tensor, image_name_list)

    def on_predict_epoch_end(self, trainer, pl_module):

        output_list = []
        ground_truths_list = []
        image_list = []
        image_name_list = []
        for batch_item in pl_module.prediction_list:
            output_list.append(batch_item["outputs"])
            ground_truths_list.append(batch_item["ground_truth"])
            image_list.append(batch_item["images"])
            image_name_list.append(batch_item["img_name"])

        output_tensor = torch.concat(output_list, dim=0)
        ground_truths_tensor = torch.concat(ground_truths_list, dim=0)

        image_name_list = [item for entry in image_name_list for item in entry]

        print(len(image_name_list))

        self._sklearn_metrics(output_tensor, ground_truths_tensor, "prediction",image_name_list)

    def _log_images(self, output_tensor, ground_truths_tensor, image_tensor,image_name_list):

        n = min(self.hyperparameters["number_of_test_img_save"], image_tensor.shape[0])  # Ensure we don't exceed available images
        image_list = [image_tensor[i] for i in range(image_tensor.shape[0])][:n]

        # Convert images to a format WandB accepts+
        processed_images = []
        for img in image_list[:n]:
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()  # Convert torch tensor to NumPy

                img = np.transpose(img, (1, 2, 0))  # Change shape from [C, H, W] to [H, W, C]
                img = (img * 255).astype(np.uint8)  # Convert to 8-bit image (if needed)
                img = Image.fromarray(img)
            if isinstance(img, np.ndarray):
                img = (img * 255).astype(np.uint8)
                img = np.transpose(img, (1, 2, 0))

                img = Image.fromarray(img)  # Convert NumPy array to PIL Image

            processed_images.append(img)


        softmax = nn.Softmax(dim=1)
        preds = softmax(output_tensor).argmax(dim=1)
        confis = softmax(output_tensor).max(dim=1).values.detach().cpu().numpy()
        y_pred = self.label_encoder.inverse_transform(preds.detach().cpu().numpy())
        y_true = self.label_encoder.inverse_transform(ground_truths_tensor.detach().cpu().numpy())

        output_serialized = [y_pred[i] for i in range(y_pred.shape[0])][:n]
        ground_truths_serialized = [y_true[i] for i in range(y_true.shape[0])][:n]
        confis_serialized  = [confis[i] for i in range(confis.shape[0])][:n]
        matches = [gt == out for gt, out in zip(ground_truths_serialized, output_serialized)]

        # Option 1: Log images with captions
        #captions = [f'Ground Truth: {gt} - Prediction: {pred} - Confidence: {con}' for gt, pred, con in zip(ground_truths_serialized, output_serialized, confis_serialized)]
        #self.logger.log_image(key='sample_images', images=processed_images, caption=captions)

        # Option 2: Log predictions as a WandB table
        columns = ['image', 'ground truth', 'prediction', 'confidence', 'match', 'img_name']
        data = [[wandb.Image(img), gt, pred, con, mat, img_name] for img, gt, pred, con, mat, img_name in
                zip(processed_images, ground_truths_serialized, output_serialized, confis_serialized, matches, image_name_list)]

        self.logger.log_table(key='UVP_1000_Match_White', columns=columns, data=data)

    def _sklearn_metrics(
        self, output: torch.Tensor, ground_truths: torch.Tensor, mode: str, img_name_list: torch.Tensor
    ):
        logger.info(("output shape", output.shape))
        logger.info(("ground_truths shape", ground_truths.shape))
        model_dir = self.hyperparameters["model_dir"]

        softmax = nn.Softmax(dim=1)
        preds = softmax(output).argmax(dim=1)
        confis = softmax(output).detach().cpu().numpy()
        y__pred_ = preds.detach().cpu().numpy()
        y__true_ = ground_truths.detach().cpu().numpy()
        y_pred = self.label_encoder.inverse_transform(y__pred_)
        y_true = self.label_encoder.inverse_transform(y__true_)

        report = classification_report(y_true, y_pred, output_dict=True)
        report_confusion_matrix = confusion_matrix(y_true, y_pred, labels=list(self.label_encoder.classes_))

        cm_normalized = confusion_matrix(y_true, y_pred, labels=list(self.label_encoder.classes_), normalize="true")

        if mode != 'prediction':
            wandb.log({"conf_mat": wandb.plot.confusion_matrix(
                                                               y_true=ground_truths.detach().cpu().numpy(), preds=preds.detach().cpu().numpy(),
                                                               class_names=list(self.label_encoder.classes_))})
        print("Evaluation metrics:")

        self.save_reports(model_dir, mode, report_confusion_matrix, report)
        self.save_test_evaluations(model_dir, mode, y_pred, y_true, confis, img_name_list)
        # Save confusion matrix plot
        try:
            self._save_confusion_matrix_plot(report_confusion_matrix, model_dir, mode)
            self._save_confusion_matrix_plot(cm_normalized, model_dir, mode+"_normalized")

        except Exception as e:
            logger.exception(e)
            logger.error("No Graphics generated")

    def _save_confusion_matrix_plot(self, cm, model_dir, mode):
        import matplotlib.pyplot as plt
        import seaborn as sns
        labels = list(self.label_encoder.classes_)
        plt.figure(figsize=(12, 12))
        if mode in ["test", "prediction"]:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        else:
            sns.heatmap(cm, annot=True,  cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix ({mode})')

        # Ensure the directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Save the figure
        cm_path = os.path.join(model_dir, f'confusion_matrix_{mode}.png')
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {cm_path}")

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
        print(f"{model_dir}/{mode}_confusion_matrix.csv")
        print("Confusion Matrix and Classication report")

    def save_test_evaluations(self, model_dir, mode, y_pred, y_true, confis, img_name_list):
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
        df_test["img_name"] = img_name_list
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
        self.model_name = self.hyperparameters["model_name"]
        if self.model_name == "resnet34":
            self.module = ResNet34(
                 hyperparameters=self.hyperparameters
            )
        elif self.model_name == "resnet50":
            self.module = ResNet50(
                 hyperparameters=self.hyperparameters
            )
        elif self.model_name == "EfficientNetB0":
            self.module = EfficientNetB0(
                hyperparameters=self.hyperparameters
            )
        else:
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
        self.prediction_list = []
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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img, ground_truth, img_name = batch
        out = self.forward(img)
        ret = {"outputs": out, "loss": None, "ground_truth": ground_truth, "images": img, "img_name": img_name}
        self.prediction_list.append(ret)
        return ret

    def predict(self, image: Union[torch.Tensor, Image.Image]):
        """Predict the class label for a given image."""
        self.eval()
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image = transform(image)
        image = image.to(self.device)
        image = torch.unsqueeze(image, 0)
        if isinstance(image, Image.Image):
            if transform:
                image = transform(image)
            image = torch.unsqueeze(image, 0)  # Add batch dimension


        logits = self.forward(image)
        softmax = nn.Softmax(dim=1)
        probs = softmax(logits)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        return pred_label, confidence
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
        img , ground_truth, img_name = batch
        out = self.forward(img)
        if mode == "train":
            loss = self.criterion(out, ground_truth)
            output = self.train_metrics(out, ground_truth)
            self.log_dict(output)
            self.log("learning_rate", self.learning_rate)
            self.log(f"{mode}_loss", loss)
            self.train_metrics.update(out, ground_truth)
        elif mode == "val":
            loss = self.criterion(out, ground_truth)
            output = self.val_metrics(out, ground_truth)
            self.log_dict(output)
            self.log(f"{mode}_loss", loss)
            self.val_metrics.update(out, ground_truth)
        elif mode == "test":
            loss = self.criterion(out, ground_truth)
            output = self.test_metrics(out, ground_truth)
            self.log_dict(output)
            self.log(f"{mode}_loss", loss)
            self.test_metrics.update(out, ground_truth)
            # reset predict list
            # self.pred_list = []





        return {"outputs": out, "loss": loss, "ground_truth": ground_truth, "images": img, "img_name": img_name}

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
        scheduler = ExponentialLR(optimizer, gamma=self.hyperparameters["gamma"])
        return [optimizer] , [{"scheduler": scheduler, "interval": "epoch"}]

    def configure_callbacks(self) -> Union[Sequence[pl.pytorch.Callback], pl.pytorch.Callback]:
        """Configure Early stopping or Model Checkpointing.

        Returns:

        """
        early_stop = EarlyStopping(
            monitor="val_MulticlassF1Score", patience=self.hyperparameters["patience"], mode="max"
        )
        checkpoint = ModelCheckpoint(
            monitor="val_MulticlassF1Score",
            mode="max",
            dirpath=self.hyperparameters["model_dir"] + "/",
            filename=self.hyperparameters["model_filename"],
            save_top_k=1,  # Save only the best model
        )

        sklearn = SklearnMetricsCallback(label_encoder=self.label_encoder, hyperparameters=self.hyperparameters, logger=self.logger)
        return [early_stop, checkpoint, sklearn]