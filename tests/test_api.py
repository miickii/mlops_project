from fastapi.testclient import TestClient
from mlops_project.api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Fruit Classifier API"}

def test_predict():
    with open("tests/test_image.jpg", "rb") as f:
        response = client.post("/predict/", files={"file": f})
    assert response.status_code == 200
    assert "predicted_class" in response.json()
    assert "confidence" in response.json()

