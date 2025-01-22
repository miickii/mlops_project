from fastapi.testclient import TestClient
import os
from google.cloud import storage

# Constants
BUCKET_NAME = "fruits_project_models"
MODEL_BLOB_PATH = "fruits_model.ckpt"
LABELS_BLOB_PATH = "classes.json"
MODEL_LOCAL_PATH = "/tmp/fruits_model.ckpt"
LABELS_LOCAL_PATH = "/tmp/classes.json"
TEST_IMAGE_PATH = "tests/test_image.jpg"

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """
    Download a file from Google Cloud Storage if it doesn't already exist locally.

    Args:
        bucket_name (str): Name of the GCS bucket.
        source_blob_name (str): Path to the blob in GCS.
        destination_file_name (str): Local destination path.
    """
    print(f"Checking if {destination_file_name} exists...")
    if not os.path.exists(destination_file_name):
        print(f"Downloading {source_blob_name} from bucket {bucket_name}...")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {source_blob_name} to {destination_file_name}")
    else:
        print(f"File already exists at {destination_file_name}")

# Ensure the model and labels are available locally
download_from_gcs(BUCKET_NAME, MODEL_BLOB_PATH, MODEL_LOCAL_PATH)
download_from_gcs(BUCKET_NAME, LABELS_BLOB_PATH, LABELS_LOCAL_PATH)

from mlops_project.api import app
client = TestClient(app)

def test_read_root():
    """
    Test the root endpoint.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Fruit Classifier API"}

def test_predict():
    """
    Test the /predict endpoint using a sample image.
    """
    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"Test image not found at {TEST_IMAGE_PATH}")

    with open("tests/test_image.jpg", "rb") as f:
        response = client.post("/predict/", files={"file": f})
    assert response.status_code == 200
    response_json = response.json()
    assert "predicted_class" in response_json
    assert "confidence" in response_json
    assert isinstance(response_json["predicted_class"], str)  # Label should be a string
    assert isinstance(response_json["confidence"], float)  # Confidence should be a float
