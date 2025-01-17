from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
import json
from mlops_project.model_lightning import FruitClassifierModel

# Initialize FastAPI app
app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load model
MODEL_PATH = "../models/fruits_model.ckpt"  # Update with your model checkpoint path
model = FruitClassifierModel.load_from_checkpoint(MODEL_PATH, lr=1e-4)
model.eval()

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

# Example usage
mapping_path = "../data/processed/classes.json"
labels = load_labels(mapping_path)
print(labels)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fruit Classifier API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the class of an uploaded image.

    Args:
        file (UploadFile): Uploaded image file.

    Returns:
        PredictionResult: Predicted class and confidence score.
    """
    try:
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

        # Return the result
        return {"predicted_class": predicted_label, "confidence": confidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
