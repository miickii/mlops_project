import torch
from pytorch_lightning import LightningModule
from mlops_project.model import ProjectModel

class FruitClassifierModel(LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = ProjectModel(num_classes=141)
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
