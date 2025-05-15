import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
import requests
import time
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()
# get the Hugging Face API token from environment variables
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Page configuration with custom styling
st.set_page_config(
    page_title="PneumoAssist: X-Ray Analysis & Healthcare Assistant",
    page_icon="🫁",
    layout="wide"
)
# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7ff;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .info-box {
        background-color: #e1efff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
    }
    .result-normal {
        background-color: #d1ffdd;
        border: 1px solid #52b788;
    }
    .result-pneumonia {
        background-color: #ffe1e1;
        border: 1px solid #e63946;
    }
    .disclaimer {
        font-size: 12px;
        color: #555;
        font-style: italic;
    }
    .chat-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        height: 400px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #e1efff;
        padding: 10px;
        border-radius: 15px 15px 0 15px;
        margin: 10px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .assistant-message {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 15px 15px 15px 0;
        margin: 10px 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    .header-section {
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)
# Application header
st.markdown("<div class='header-section'><h1>🫁 PneumoAssist: X-Ray Analysis & Medical Assistant</h1></div>", unsafe_allow_html=True)
# Load pneumonia model with caching for efficiency
@st.cache_resource
def load_pneumonia_model():
    try:
        model = load_model('cnn_model91.h5')
        return model
    except Exception as e:
        st.error(f"Error loading pneumonia model: {str(e)}")
        return None
# Load model
pneumonia_model = load_pneumonia_model()
# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial greeting message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm PneumoAssist, your medical assistant for pneumonia detection and information. I can analyze chest X-rays and answer your questions about pneumonia. How can I help you today?"
    })
# Image processing function with better error handling
def process_image(uploaded_file):
    try:
        # Open image
        image = Image.open(uploaded_file)
        # Keep a copy of the original for display
        display_image = image.copy()
        # Process for model
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((150, 150))  # Resize to model input size
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array, display_image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None
    
# Function to handle the chat conversation using Hugging Face Inference APIimport requests
def chat_response(user_input):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"  # Note: v0.1 is more stable
    
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    
    # Key Fix: Use INST/<<SYS>> format that Mistral-7B-Instruct expects
    prompt = f"""<s>[INST] <<SYS>>
You are PneumoAssist, a concise assistant. You are also able to predict pneumonia based on x ray images with a 91% accuracy when the user submits an image. 
Respond briefly and shortly to the user's input. Do NOT continue conversations or assume symptoms or anything.
<</SYS>>

User: {user_input} [/INST]"""
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,  # Shorter responses
            "temperature": 0.1,    # More deterministic
            "do_sample": False,
            "repetition_penalty": 1.5  # Avoid repetition
        }
    }
    

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    full_text = response.json()[0]['generated_text']
    # Isolate the response after [/INST]
    return full_text.split("[/INST]")[-1].strip()

    
# Function to analyze the image and provide medical insights
def analyze_image(uploaded_file):
    with st.spinner("Analyzing your X-ray image..."):
        processed_image, original_image = process_image(uploaded_file)
        if processed_image is not None and pneumonia_model is not None:
            # Make prediction
            prediction = pneumonia_model.predict(processed_image)[0][0]
            probability_percentage = prediction * 100
            # Display the prediction result
            if prediction > 0.5:
                result_class = "Potential Pneumonia Detected"
                result_color = "result-pneumonia"
                explanation = f"""
                The analysis indicates features that may suggest pneumonia, with a confidence level of {probability_percentage:.1f}%.
                Common pneumonia indicators in X-rays include:
                - Areas of increased opacity (white patches)
                - Consolidation in lung fields
                - Potential air bronchograms
                Important: This is an automated screening tool and not a definitive diagnosis. Please consult with a healthcare professional immediately for proper evaluation and treatment.
                """
            else:
                result_class = "No Clear Signs of Pneumonia"
                result_color = "result-normal"
                explanation = f"""
                The analysis does not show clear indications of pneumonia (confidence: {100-probability_percentage:.1f}% normal).
                The lung fields appear to have expected transparency without significant areas of consolidation or opacities typically associated with pneumonia.
                Note: If you're experiencing symptoms, please consult a healthcare provider regardless of this result.
                """
            # Display original image and results
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(original_image, caption="Uploaded X-ray", use_container_width=True)
            with col2:
                st.info("Analysis Visualization Coming Soon...", icon="💡") # Placeholder for future visualization
            # Display result box with properly escaped HTML
            cleaned_explanation = explanation.replace('\n', '<br>').strip()
            st.markdown(f"""
            <div class="result-box {result_color}">
                <h3>{result_class}</h3>
                <div>{cleaned_explanation}</div>
                <div class="disclaimer">
                    Medical Disclaimer: This tool is for educational and screening purposes only.
                    It is not a substitute for professional medical advice, diagnosis, or treatment.
                </div>
            </div>
            """, unsafe_allow_html=True)
            # Log the analysis in chat history - clean up the content
            analysis_summary = f"I've analyzed your X-ray. {result_class} (Confidence: {probability_percentage:.1f}%)."
            # Only add to messages if not already there (prevent duplication)
            if not any(msg["content"] == analysis_summary and msg["role"] == "assistant" for msg in st.session_state.messages[-3:]):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": analysis_summary
                })
            return True
    return False
# Create two columns for layout
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("<h3>Upload X-ray for Analysis</h3>", unsafe_allow_html=True)
    # Display info box
    st.markdown("""
    <div class="info-box">
        <strong>How to use:</strong>
        <ol>
            <li>Upload a chest X-ray image (PA or AP view)</li>
            <li>Wait for the analysis results</li>
            <li>Ask follow-up questions in the chat</li>
        </ol>
        <p class="disclaimer">This tool analyzes chest X-rays for potential signs of pneumonia with ~90% accuracy on test datasets. Always consult healthcare professionals for diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)
    # File uploader with key to track changes
    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"], key="file_uploader")
    # Track if we've processed this file already
    if "last_processed_file" not in st.session_state:
        st.session_state.last_processed_file = None
    if uploaded_file is not None and uploaded_file != st.session_state.last_processed_file:
        # Update the last processed file
        st.session_state.last_processed_file = uploaded_file
        # Log the upload in chat history
        st.session_state.messages.append({"role": "user", "content": "I've uploaded a chest X-ray for analysis."})
        # Process the image
        analyze_image(uploaded_file)
with col2:
    st.markdown("<h3>Healthcare Assistant Chat</h3>", unsafe_allow_html=True)
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
    # Chat input with a submit button to prevent looping
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask me about pneumonia or the X-ray analysis...", key="user_input")
        submit_button = st.form_submit_button("Send")
        if submit_button and user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            # Get bot response
            response = chat_response(user_input)
            # Add bot response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Use rerun to update the display
            st.rerun()
# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
    <p class="disclaimer">PneumoAssist is an educational tool and should not replace professional medical advice.
    The X-ray analysis is performed using a convolutional neural network trained on chest X-ray datasets.
    Always consult with qualified healthcare professionals for proper diagnosis and treatment.</p>
</div>
""", unsafe_allow_html=True)