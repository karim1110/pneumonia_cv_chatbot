import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
import requests
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

st.set_page_config(
    page_title="PneumoAssist: X-Ray Analysis & Healthcare Assistant",
    page_icon="🫁",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f5f7ff; }
    .stApp { max-width: 1200px; margin: 0 auto; }
    .info-box { background-color: #e1efff; border-radius: 10px; padding: 15px; margin-bottom: 15px; }
    .result-box { padding: 20px; border-radius: 10px; margin-top: 15px; }
    .result-normal { background-color: #d1ffdd; border: 1px solid #52b788; }
    .result-pneumonia { background-color: #ffe1e1; border: 1px solid #e63946; }
    .disclaimer { font-size: 12px; color: #555; font-style: italic; }
    .chat-container { border: 1px solid #ddd; border-radius: 10px; padding: 20px; background-color: white; height: 400px; overflow-y: auto; }
    .user-message { background-color: #e1efff; padding: 10px; border-radius: 15px 15px 0 15px; margin: 10px 0; max-width: 80%; float: right; clear: both; }
    .assistant-message { background-color: #f0f0f0; padding: 10px; border-radius: 15px 15px 15px 0; margin: 10px 0; max-width: 80%; float: left; clear: both; }
    .header-section { text-align: center; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header-section'><h1>🫁 PneumoAssist: X-Ray Analysis & Medical Assistant</h1></div>", unsafe_allow_html=True)

@st.cache_resource
def load_pneumonia_model():
    try:
        model = load_model('cnn_model91.h5')
        return model
    except Exception as e:
        st.error(f"Error loading pneumonia model: {str(e)}")
        return None

pneumonia_model = load_pneumonia_model()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": ("Hello! I'm PneumoAssist, your medical assistant for pneumonia detection and information. "
                    "I can analyze chest X-rays and answer your questions about pneumonia. "
                    "Please note that I do not provide diagnostic scores or probabilities. "
                    "My responses include important limitations and precautions about healthcare. How can I assist you?")
    })

def process_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        display_image = image.copy()
        image = image.convert('L')
        image = image.resize((150, 150))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array, display_image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def chat_response(user_input):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    history = st.session_state.messages[-5:-1] if len(st.session_state.messages) > 1 else []
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

    prompt = f"""<s>[INST] <<SYS>>
You are PneumoAssist, a concise assistant specialized in pneumonia detection and healthcare advice.
Do NOT provide any numeric scores or confidence levels in your replies.
Always include clear healthcare limitations and precautions.
Keep responses short and factual, avoid speculation beyond the immediate question.
<</SYS>>

{history_str}
User: {user_input} [/INST]"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.1,
            "do_sample": False,
            "repetition_penalty": 1.5
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    full_text = response.json()[0]['generated_text']
    reply = full_text.split("[/INST]")[-1].strip()

    # Remove any numeric scores if present (extra precaution)
    import re
    reply = re.sub(r'(\d+(\.\d+)?%?)', '', reply)  
    # Add healthcare limitation reminder if missing
    if "consult a healthcare professional" not in reply.lower():
        reply += "\n\nPlease remember this tool does not replace professional medical advice. Always consult a healthcare provider."

    return reply.strip()

def analyze_image(uploaded_file):
    with st.spinner("Analyzing your X-ray image..."):
        processed_image, original_image = process_image(uploaded_file)
        if processed_image is not None and pneumonia_model is not None:
            prediction = pneumonia_model.predict(processed_image)[0][0]
            # For UI, keep showing the result with explanations but no numeric scores in chat
            if prediction > 0.5:
                result_class = "Potential Pneumonia Detected"
                result_color = "result-pneumonia"
                explanation = """
                The X-ray analysis suggests signs that may be consistent with pneumonia, such as increased opacity or consolidation.
                IMPORTANT: This is an automated screening tool only and NOT a diagnosis.
                Please seek immediate evaluation from a qualified healthcare professional for accurate diagnosis and treatment.
                """
            else:
                result_class = "No Clear Signs of Pneumonia"
                result_color = "result-normal"
                explanation = """
                The X-ray analysis does not show clear signs typically associated with pneumonia.
                However, if you have symptoms, please consult a healthcare provider regardless of this automated result.
                """

            st.image(original_image, caption="Uploaded X-ray", use_container_width=True)

            cleaned_explanation = explanation.replace('\n', '<br>').strip()
            st.markdown(f"""
            <div class="result-box {result_color}">
                <h3>{result_class}</h3>
                <div>{cleaned_explanation}</div>
                <div class="disclaimer">
                    Medical Disclaimer: This tool is for educational and screening purposes only.
                    It is NOT a substitute for professional medical advice, diagnosis, or treatment.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Append a very brief assistant message without scores or probabilities
            analysis_summary = result_class + ". Please remember this is not a diagnostic tool."

            st.session_state.messages.append({
                "role": "assistant",
                "content": analysis_summary
            })
            return True
    return False

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("<h3>Upload X-ray for Analysis</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>How to use:</strong>
        <ol>
            <li>Upload a chest X-ray image (PA or AP view)</li>
            <li>Wait for the analysis results</li>
            <li>Ask follow-up questions in the chat</li>
        </ol>
        <p class="disclaimer">This tool analyzes chest X-rays for potential pneumonia signs using an AI model. It is not a diagnostic device. Always consult healthcare professionals for medical decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"], key="file_uploader")
    if "last_processed_file" not in st.session_state:
        st.session_state.last_processed_file = None
    if uploaded_file is not None and uploaded_file != st.session_state.last_processed_file:
        st.session_state.last_processed_file = uploaded_file
        st.session_state.messages.append({"role": "user", "content": "I've uploaded a chest X-ray for analysis."})
        analyze_image(uploaded_file)

with col2:
    st.markdown("<h3>Healthcare Assistant Chat</h3>", unsafe_allow_html=True)
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask me about pneumonia or the X-ray analysis...", key="user_input")
        submit_button = st.form_submit_button("Send")
        if submit_button and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = chat_response(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()

st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
    <p class="disclaimer">
    PneumoAssist is an educational tool and should NOT replace professional medical advice. 
    The X-ray analysis is performed using an AI model trained on chest X-ray datasets.
    Always consult with qualified healthcare professionals for diagnosis and treatment.
    Use this tool with caution and understand its limitations in healthcare.
    </p>
</div>
""", unsafe_allow_html=True)
