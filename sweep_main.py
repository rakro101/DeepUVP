# https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1

from lightning.pytorch.loggers import WandbLogger
import wandb
import lightning as pl
from dataloader import ImgDataModule
from loop import LitModel
from config import HYPERPARAMETERS


def sweep_iteration():
    # set up W&B logger
    wandb.init()  # required to have access to `wandb.config`
    wandb_logger = WandbLogger()
    # setup data
    config = wandb.config
    hyperparameters = HYPERPARAMETERS
    hyperparameters.update({"learning_rate": config.lr})
    hyperparameters.update(
        {"num_freeze_layers_resnet": config.num_freeze_layers_resnet}
    )
    # hyperparameters.update({"batch_size": config.batch_size})

    data_module = ImgDataModule(hyperparameters=hyperparameters)
    # setup model - note how we refer to sweep parameters with wandb.config
    model = LitModel(
        hyperparameters=hyperparameters,
    )

    # setup Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=hyperparameters["max_epochs"],
        accelerator="gpu",
        devices=hyperparameters["devices"],
        # strategy=DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else "auto",
        precision=hyperparameters["precision"],
        limit_train_batches=hyperparameters["limit_batches"],
        limit_test_batches=hyperparameters["limit_batches"],
        limit_val_batches=hyperparameters["limit_batches"],
        log_every_n_steps=1,
        # fast_dev_run=True,
    )
    # train

    trainer.fit(model, data_module)
    if hyperparameters["val_mode"] == "on":
        print("Eval on validation set")
        trainer.validate(
            model,
            data_module,
            ckpt_path="best",
        )

    if hyperparameters["test_mode"] == "on":
        print("Testing on test set")
        trainer.test(
            model,
            data_module,
        )


sweep_config = {
    "run_cap": 10,
    "method": "bayes",  # Random search
    "metric": {  # We want to maximize val_acc
        "name": "val_MulticlassF1Score",
        "goal": "maximize",
    },
    "early_terminate": {"type": "hyperband", "min_iter": 3},
    "parameters": {
        "lr": {
            # log uniform distribution between exp(min) and exp(max)
            "distribution": "log_uniform",
            "min": -9.21,  # exp(-9.21) = 1e-4
            "max": -4.61,  # exp(-4.61) = 1e-2
        },
        "num_freeze_layers_resnet": {"values": [5, 7, 9]},
        # "batch_size": {"values": [512, 1024]},
    },
}

if __name__ == "__main__":
    # wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, project="DeepUVP_", entity="hhuml"
    )  # if not in hhuml obmit
    wandb.agent(sweep_id, function=sweep_iteration, count=10)
    wandb.finish()
