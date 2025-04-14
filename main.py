"""Train, validate and test the model."""

import logging
import os

import lightning as pl
import torch
from dataloader import ImgDataModule
from lightning.pytorch.strategies import DDPStrategy
from loop import LitModel
from lightning.pytorch.loggers import WandbLogger
import wandb

wandb.login()
logger = logging.getLogger(__name__)


def lightning_training(model_dir: str, hyperparameters: dict) -> object:
    """Executes the training process. This involves creating the dataset and corresponding dataloader, initializing the
    model, and then training, validating, and testing the model.

    Args:
        model_dir (str): The path where the model output will be saved.
        hyperparameters (dict): A dictionary containing the hyperparameters for training.

    Returns:
        model: The trained model.

    """
    logger.debug(f"hyperparameters: {hyperparameters}, {type(hyperparameters)}")
    os.makedirs("lightning_logs", exist_ok=True)
    data_module = ImgDataModule(hyperparameters=hyperparameters)
    number_classes = hyperparameters["num_classes"]
    logger.info("Limit batches %s" % hyperparameters["limit_batches"])
    logger.debug("num_classes %s" % number_classes)

    model = LitModel(
        hyperparameters=hyperparameters,
    )

    wandb_logger = WandbLogger(
        log_model="all", project="DeepUVP", entity="hhu-marine"
    )  # if not in hhuml obmit

    print(wandb_logger.experiment.name)
    print(wandb_logger.experiment.path)
    print(wandb_logger.experiment.id)
    print(wandb_logger.experiment._run_id)
    print(wandb_logger.experiment.entity)
    print(wandb_logger.experiment.dir)
    hyperparameters.update({"run_name": wandb_logger.experiment.name})
    hyperparameters.update({"run_id": wandb_logger.experiment.dir})
    hyperparameters.update({"model_dir": wandb_logger.experiment.dir})

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=hyperparameters["max_epochs"],
        accelerator="gpu",
        devices=hyperparameters["devices"],
        default_root_dir=model_dir,
        # strategy=DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else "auto",
        precision=hyperparameters["precision"],
        limit_train_batches=hyperparameters["limit_batches"],
        limit_test_batches=hyperparameters["limit_batches"],
        limit_val_batches=hyperparameters["limit_batches"],
        log_every_n_steps=1,
        # fast_dev_run=True,
    )
    trainer.fit(model, data_module)
    logger.debug("trainer model %s" % trainer.model)
    try:
        trainer.save_checkpoint("trained_model", weights_only=False)
    except Exception as e:
        logger.error(f"{e}")

    if hyperparameters["val_mode"] == "on":
        logger.info("Validate Model")
        logger.debug("trainer_val model %s" % trainer.model)
        try:
            trainer.validate(
                model,
                data_module,
                ckpt_path="best",
            )
        except Exception as e:
            logger.error(f"{e}")
    if hyperparameters["test_mode"] == "on":
        logger.info("Test Model")
        logger.debug("trainer_test model %s" % trainer.model)
        try:
            trainer.test(
                model,
                data_module,
                ckpt_path="best",
            )
        except Exception as e:
            logger.error(f"{e}")

    wandb.finish()
    return trainer


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    from config import HYPERPARAMETERS

    print(HYPERPARAMETERS)
    lightning_training(model_dir="logs", hyperparameters=HYPERPARAMETERS)
