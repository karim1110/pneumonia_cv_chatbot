import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="Pneumonia Detection Chatbot", page_icon="🩺")

@st.cache_resource
def load_pneumonia_model():
    model = load_model('cnn_model91.h5')
    return model

model = load_pneumonia_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can help analyze chest X-rays for signs of pneumonia. Please upload an image and I'll give you my assessment."}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Image upload and processing
def process_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        image = image.convert('RGB')
        image = image.resize((150, 150))  # Adjust size based on the model's requirements
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Chat input
if prompt := st.chat_input("Ask me about pneumonia or upload an image..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if the user uploaded a file
    if "upload" in prompt.lower() or "image" in prompt.lower():
        response = "Please upload a chest X-ray image using the file uploader below."
    else:
        response = "I can analyze chest X-rays for signs of pneumonia. Please upload an image for assessment."
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# File uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    # Process the image
    processed_image = process_image(uploaded_file)
    
    if processed_image is not None and model is not None:
        # Get prediction (0 = Normal, 1 = Pneumonia)
        prediction = model(processed_image)
        
        # Generate appropriate response
        if prediction == 1:  # Pneumonia detected
            response = """
            I've detected something unusual in your chest X-ray that may indicate pneumonia. 
            While this analysis isn't definitive, I would strongly recommend you:
            - Visit a healthcare professional for further evaluation
            - Get proper diagnostic testing
            - Follow up with appropriate treatment if needed
            
            It's always better to be cautious with respiratory symptoms.
            """
        else:  # Normal
            response = """
            I don't detect clear signs of pneumonia in your X-ray. However:
            - If you're experiencing symptoms like fever, cough, or difficulty breathing
            - Or if you're feeling unwell
            
            It might still be worth consulting with a healthcare provider for a complete evaluation.
            """
        
        # Add the prediction to chat history
        st.session_state.messages.append({"role": "user", "content": "Here's my X-ray image"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display in chat interface
        with st.chat_message("user"):
            st.markdown("Here's my X-ray image")
        with st.chat_message("assistant"):
            st.markdown(response)

# Add a disclaimer
st.sidebar.markdown("""
**Disclaimer:**
This tool is for informational purposes only and not a substitute for professional medical advice. 
Always consult with a qualified healthcare provider for medical concerns.
""")