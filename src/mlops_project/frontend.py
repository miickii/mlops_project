
import os

import pandas as pd
import requests
import streamlit as st

def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/predict/"
    response = requests.post(predict_url, files={"file": image}, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None

def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = "https://fruits-model-api-jelcai3tmq-ew.a.run.app"
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)
        print(result)

        if result is not None:
            prediction = result["predicted_class"]
            confidence = result["confidence"]

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", prediction)
            st.write("Confidence:", f"{confidence * 100:.2f}%")

        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
