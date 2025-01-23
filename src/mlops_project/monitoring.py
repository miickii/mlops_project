import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from google.cloud import storage
import json
import os

# GCS Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "your-gcs-bucket-name")
GCS_CLIENT = storage.Client()

def fetch_predictions(bucket_name: str, n: int = 1000) -> pd.DataFrame:
    """Fetch the latest predictions from GCS."""
    bucket = GCS_CLIENT.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix="predictions/")
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
    bucket = GCS_CLIENT.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(report.to_json())

if __name__ == "__main__":
    # Load reference data (e.g., training data)
    reference_data = pd.read_csv("data/processed/train_targets.csv")  # Adjust path as needed

    # Fetch current predictions
    current_data = fetch_predictions(GCS_BUCKET_NAME, n=1000)

    # Generate drift report
    report = generate_drift_report(reference_data, current_data)

    # Save report to GCS
    save_report_to_gcs(GCS_BUCKET_NAME, report, "drift_reports/latest_report.json")