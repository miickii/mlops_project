import os
import requests
import streamlit as st
from io import BytesIO

def classify_image(image_bytes, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/predict/"
    # Simulate a file-like object with a name
    files = {"file": ("image.jpg", BytesIO(image_bytes), "image/jpeg")}
    response = requests.post(predict_url, files=files, timeout=10)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")  # For debugging purposes
        return None

def main() -> None:
    """Main function of the Streamlit frontend."""
    # Set page title/icon (optional)
    st.set_page_config(page_title="Fruits Classifier", page_icon="🍓")

    # Define your backend URL
    backend = "https://mlops-fruits-api-10603713134.europe-west1.run.app/"
    if backend is None:
        raise ValueError("Backend service not found")

    # Title and description
    st.title("Fruit Image Classification 🍎🍌🥝")
    st.write("This app uses a machine learning model to identify fruit in images. \
              Simply upload a fruit image, and get the predicted fruit name along with confidence.")

    # Use a side column (sidebar) for upload if you want
    st.sidebar.title("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # If file is uploaded, proceed with classification
    if uploaded_file is not None:
        # Convert the uploaded file to bytes
        image_bytes = uploaded_file.read()

        # Optional: show the image in the main area using columns
        col1, col2 = st.columns(2)
        with col1:
            # Show the uploaded image in the first column
            st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

        # Classify the image
        result = classify_image(image_bytes, backend=backend)

        if result is not None:
            prediction = result["predicted_class"]
            confidence = float(result["confidence"])

            # Show results in the second column
            with col2:
                st.subheader("Classification Results")
                st.write(f"**Prediction:** {prediction}")
                
                # Display confidence as a progress bar or as a metric
                st.progress(confidence)  # progress bar with range [0..1]
                st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")

        else:
            st.error("Failed to get a prediction from the backend.")
    else:
        # If no file is uploaded, show a placeholder message in the main section
        st.info("Please upload an image to classify.")

if __name__ == "__main__":
    main()
