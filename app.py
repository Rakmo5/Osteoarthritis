import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image

# ✅ Fix bounded ReLU
def bounded_relu(x):
    return tf.keras.activations.relu(x, max_value=1.0)

# ✅ Load model with fixed activation function
@st.cache_resource
def load_trained_model():
    custom_objects = {"bounded_relu": bounded_relu}
    model = tf.keras.models.load_model("models/model_4.keras", custom_objects=custom_objects)
    return model

model = load_trained_model()

# ✅ Streamlit UI
st.title("Arthritis X-ray Classification")

uploaded_file = st.file_uploader("Upload an X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # ✅ Preprocess image
    image = image.resize((224, 224))  # Adjust size to match model input
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # ✅ Predict
    prediction = model.predict(image_array)
    import numpy as np

    # Convert prediction output to a NumPy array
    prediction = np.array(prediction, dtype=object)  # Use dtype=object if shape is inconsistent

    print(f"Prediction Shape: {prediction.shape}")  # Checking shape

    print(f"Raw Prediction Output: {prediction}")  # Debugging actual output
    print(f"Prediction Type: {type(prediction)}")  # Checking data type
    print(f"Prediction Shape: {np.shape(prediction)}")  # Checking shape

    # Now apply argmax
    predicted_class = np.argmax(prediction, axis=-1)[0]  # Use last axis for multi-class

    st.write(f"**Predicted Arthritis Severity Grade:** {predicted_class}")
