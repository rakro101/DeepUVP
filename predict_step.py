from dataloader import ImgDataModule
from config import HYPERPARAMETERS
from loop import LitModel
import lightning as pl

# Todo Atm the predict dataset has labels


def load_model(checkpoint_path, device="mps"):
    model = LitModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model, device


if __name__ == "__main__":
    model_path = "wandb/run-20250326_162954-l87fmdvx/files/best.ckpt"

    data_loader = ImgDataModule(hyperparameters=HYPERPARAMETERS)
    model, device = load_model(model_path)
    trainer = pl.Trainer(precision=HYPERPARAMETERS["precision"])
    predictions = trainer.predict(model, data_loader)
