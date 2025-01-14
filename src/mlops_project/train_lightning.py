import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from mlops_project.dataset import get_dataloaders
from mlops_project.model import ProjectModel
import typer
from pytorch_lightning import LightningModule
import wandb

app = typer.Typer()

class FruitClassifierModule(LightningModule):
    def __init__(self, model, num_classes, lr):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


@app.command()
def train(
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 1e-4,
    model_name: str = "fruits_model.pth",
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

    train_loader, _ = get_dataloaders(batch_size=batch_size, transform=None)

    # Define model
    num_classes = 141
    base_model = ProjectModel(num_classes=num_classes)
    model = FruitClassifierModule(base_model, num_classes, lr)

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
