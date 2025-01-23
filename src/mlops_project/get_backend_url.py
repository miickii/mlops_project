from google.cloud import run_v2 # pip install google-cloud-run streamlit
import os

# Set the backend URL
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/dtumlops-447821/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "fruits-model-api":
            return service.uri
    return os.environ.get("BACKEND", None)

if __name__ == "__main__":
    print(get_backend_url())