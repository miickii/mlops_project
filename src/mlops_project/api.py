from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from io import BytesIO
from PIL import Image
import torch
from pytorch_lightning import LightningModule
import torchvision.transforms as transforms
import json
import timm
from torch import nn
from pathlib import Path
import os
from google.cloud import storage
from datetime import datetime
import uuid

def ProjectModel(num_classes: int, pretrained:bool = True):
    """
    Create and configure the model.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use a pretrained model.

    Returns:
        torch.nn.Module: Configured model.
    """
    if num_classes <= 0:
        raise ValueError("Number of classes must be greater than zero.")

    # Load pretrained ResNet
    model = timm.create_model("resnet18", pretrained=pretrained)

    # Modify the final layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

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


# Initialize FastAPI app
app = FastAPI()

# GCS Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "mlops-predictions")  # Replace with your bucket name
GCS_CLIENT = storage.Client()

def upload_to_gcs(bucket_name: str, file_name: str, data: bytes):
    """Upload data to a GCS bucket."""
    bucket = GCS_CLIENT.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(data)

def log_prediction(image_bytes: bytes, prediction: dict):
    """Log prediction data to GCS."""
    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"predictions/{timestamp}_{uuid.uuid4().hex}.json"

    # Save prediction data as JSON
    prediction_data = {
        "timestamp": timestamp,
        "predicted_class": prediction["predicted_class"],
        "confidence": prediction["confidence"],
        "image_size": len(image_bytes),
    }
    upload_to_gcs(GCS_BUCKET_NAME, file_name, json.dumps(prediction_data).encode())

# Device configuration
DEVICE = torch.device("cpu")  # Default to CPU since Cloud Run uses CPUs

# Define paths to model and labels
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust based on project structure
MODEL_PATH = BASE_DIR / "models/fruits_model.ckpt"
LABELS_PATH = BASE_DIR / "data/processed/classes.json"

MODEL_PATH = os.getenv("MODEL_PATH", MODEL_PATH)
LABELS_PATH = os.getenv("LABELS_PATH", LABELS_PATH)

# Validate model and labels paths
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

if not Path(LABELS_PATH).exists():
    raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}")

print(f"Loading model from: {MODEL_PATH}")
print(f"Loading labels from: {LABELS_PATH}")

# Load model
model = FruitClassifierModel.load_from_checkpoint(MODEL_PATH, lr=1e-4, map_location=torch.device("cpu"))
model.to(DEVICE)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def load_labels(mapping_path: str) -> dict:
    """Load class-to-index mapping."""
    with open(mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # Reverse the dictionary
    return idx_to_class

# Load labels
labels = load_labels(LABELS_PATH)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fruit Classifier API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        if file.size > 5 * 1024 * 1024:  # 5 MB limit
            raise HTTPException(status_code=400, detail="File too large")

        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Apply transformations
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_index = probabilities.argmax(dim=1).item()
            confidence = probabilities.max().item()

        predicted_label = labels[predicted_index]
        prediction_result = {"predicted_class": predicted_label, "confidence": confidence}

        # Log prediction in the background
        if background_tasks:
            background_tasks.add_task(log_prediction, image_bytes, prediction_result)

        return prediction_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")