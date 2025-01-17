import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from mlops_project.dataset import FruitsDataset
from mlops_project.model_lightning import FruitClassifierModel
import typer
import wandb
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get WandB API key from the environment variable
wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key is None:
    raise ValueError("WANDB_API_KEY not found in environment variables. Please set it in the .env file.")

# Log in to WandB
wandb.login(key=wandb_api_key)

app = typer.Typer()

@app.command()
def train(
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-4,
    model_name: str = "fruits_model",
    project_name: str = "fruits_classification",
):
    """Train a fruit classification model with PyTorch Lightning."""
    # Seed everything for reproducibility
    #seed_everything(42)

    # Ensure any previous W&B runs are finalized
    wandb.finish()

    # Initialize W&B Logger
    wandb_logger = WandbLogger(project=project_name, log_model=True, reinit=True)
    print(f"Starting a new W&B run in project: {project_name}")

    # Initialize dataset and DataLoader
    train_dataset = FruitsDataset(data_folder="data/processed", train=True)
    train_loader = train_dataset.get_dataloader(batch_size=batch_size)

    # Define model
    model = FruitClassifierModel(lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", filename=model_name, monitor="train_loss", mode="min", save_top_k=1
    )
    early_stopping_callback = EarlyStopping(monitor="train_loss", patience=3, mode="min")

    # Trainer
    trainer = Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    app()
