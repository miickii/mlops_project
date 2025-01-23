import torch
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from google.cloud import storage
import json
import os

# GCS Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "mlops-predictions")
GCS_CLIENT = storage.Client()

def fetch_predictions(bucket_name: str, n: int = 1000) -> pd.DataFrame:
    """Fetch the latest predictions from GCS."""
    bucket = GCS_CLIENT.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix="predictions/")
    print(blobs)
    predictions = []

    for blob in blobs:
        if len(predictions) >= n:
            break
        data = json.loads(blob.download_as_text())
        predictions.append(data)

    return pd.DataFrame(predictions)

def generate_drift_report(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    """Generate a drift report using Evidently."""
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    return report

def save_report_to_gcs(bucket_name: str, report, file_name: str):
    """Save the drift report to GCS."""
    # Save the report as a JSON file locally
    local_json_path = os.path.join(os.getenv("TEMP"), "report.json")
    report.save_json(local_json_path)
    
    # Read the saved JSON file
    with open(local_json_path, "r") as f:
        report_json = f.read()

    # Upload the JSON file contents to GCS
    bucket = GCS_CLIENT.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(report_json, content_type="application/json")

def load_labels(mapping_path: str) -> dict:
    """Load class-to-index mapping."""
    with open(mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    return class_to_idx

if __name__ == "__main__":
    # Load labels mapping
    LABELS_PATH = "data/processed/classes.json"  # Adjust path as needed
    labels = load_labels(LABELS_PATH)

    # Load reference data from PyTorch tensor
    train_targets_tensor = torch.load("data/processed/train_targets.pt")  # Adjust path as needed

    # Convert indices to class labels
    train_labels = train_targets_tensor.numpy()

    # Create a reference DataFrame
    reference_data = pd.DataFrame({
        "target": train_labels
    })

    # Fetch current predictions
    current_data = fetch_predictions(GCS_BUCKET_NAME, n=1000)

    # Map indices in current predictions to labels if needed
    if "predicted_class" in current_data.columns:
        current_data["predicted_class"] = current_data["predicted_class"].map(labels)
    
    current_data = current_data.rename(columns={"predicted_class": "target"})

    print(reference_data.head(), current_data.head())
    # Generate drift report
    report = generate_drift_report(reference_data, current_data)

    # Save report to GCS
    save_report_to_gcs(GCS_BUCKET_NAME, report, "drift_reports/latest_report.json")