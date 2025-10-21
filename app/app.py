import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
import pathlib

# --- NEW LIBRARIES TO IMPORT ---
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# Your original commented-out CSS loader
# def load_css(css_path):
#     # css_path can be a str or pathlib.Path
#     path = str(css_path)
#     with open(path) as f:
#         st.html(f"<style>{f.read()}</style>")
# css_path = pathlib.Path(__file__).parent / "styles.css"
# load_css(css_path)


# --- NEW FUNCTION 1: LOAD THE PRE-TRAINED MODEL ---
@st.cache_resource # Cache the model so it doesn't reload on every interaction
def load_model_and_processor():
    """Loads the pre-trained deepfake detection model and its processor."""
    # This model is specifically fine-tuned for classifying real vs. AI-generated images
    model_name = "dima806/deepfake_vs_real_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

# --- NEW FUNCTION 2: PERFORM DEEPFAKE DETECTION ---
def detect_deepfake(image_to_check):
    """
    Detects deepfakes using a pre-trained Vision Transformer model.
    Returns the probability that the image is fake.
    """
    processor, model = load_model_and_processor()
    
    # The processor prepares the image to be compatible with the model
    inputs = processor(images=image_to_check, return_tensors="pt")

    # Run the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Convert model's raw output (logits) into probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # The model labels are {0: 'real', 1: 'fake'}. We want the probability for 'fake'.
    fake_probability = probabilities[0][1].item()
    
    return fake_probability

# Your original ELA function
def perform_ela(uploaded_file_bytes, quality=90):
    original_image = Image.open(uploaded_file_bytes).convert('RGB')
    temp_buffer = io.BytesIO()
    original_image.save(temp_buffer, format='JPEG', quality=quality)
    temp_buffer.seek(0)
    recompressed_image = Image.open(temp_buffer)
    ela_image = ImageChops.difference(original_image, recompressed_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image


# --- UI PART (Your original code) ---
st.header("Welcome to AUTHENTISIGHT!")
st.markdown("---")
st.write("Your essential web tool for verifying the authenticity of digital images. Using advanced computer vision and deep learning algorithms, it swiftly analyzes uploaded photos to determine the likelihood of them being AI-generated (synthetic media) or digitally morphed/altered (deepfakes)")

uploaded_file=st.file_uploader("Upload an Image for Analysis", type=["jpg", "jpeg", "png"])

st.sidebar.title("About")
st.sidebar.write("Home")
st.sidebar.write("Contact")
st.sidebar.header("Configuration")
ela_quality = st.sidebar.slider("ELA Compression Quality:(Default is 90)", 70, 100, 90)

analysis_mode = st.sidebar.radio("Select Analysis Type:", ['Deepfake Detection', 'Traditional Forgery (Splicing/Copy-Move)'])
st.sidebar.info("Analysis Model: Loaded and ready for processing.")


if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Image to Analyze', use_container_width=True)

    # --- MODIFIED LOGIC: Unified button for both analysis types ---
    if st.button('Run Analysis'):
        
        # --- LOGIC FOR DEEPFAKE DETECTION ---
        if analysis_mode == 'Deepfake Detection':
            with st.spinner('Running AI deepfake analysis... This may take a moment.'):
                # We need to open the image with PIL for the model
                image_for_model = Image.open(uploaded_file).convert("RGB")
                fake_probability = detect_deepfake(image_for_model)
            
            st.success("✅ Analysis Complete!")
            st.subheader("Deepfake Analysis Results")
            st.metric(label="AI-Generated Probability Score", value=f"{fake_probability:.2%}")

            DEEPFAKE_THRESHOLD = 0.65  # 65% confidence threshold
            if fake_probability > DEEPFAKE_THRESHOLD:
                st.error("⚠️ Warning: This image is likely AI-generated or a deepfake.")
            else:
                st.success("✅ The image appears to be authentic.")
        
        # --- LOGIC FOR TRADITIONAL FORGERY (ELA) ---
        else: # This block runs for 'Traditional Forgery'
            with st.spinner('Processing... Generating Error Level Analysis (ELA) Image.'):
                result_ela_image = perform_ela(uploaded_file, quality=ela_quality)

            st.success("✅ Analysis Complete!")
            st.subheader("ELA Visualization (Bright areas are potential tampered regions)")
            st.image(result_ela_image, caption='Error level analysis Map', use_container_width=True)

            ela_array = np.array(result_ela_image.convert('L'))
            std_dev = np.std(ela_array)
            TAMPER_THRESHOLD = 20
            
            st.metric(label="Tampering Score (Standard Deviation of ELA)", value=f"{std_dev:.2f}")
            
            st.subheader("Final Verdict:")
            if std_dev > TAMPER_THRESHOLD:
                st.error("⚠️ Warning: The image shows signs of potential tampering or manipulation.")
            else:
                st.success("✅ The image appears to be authentic with no significant signs of tampering.")
            st.write("Error Level Analysis and noise analysis indicate uniform compression and noise patterns across the image.")

# --- NOTE: Removed the duplicated ELA code that ran automatically ---
# The logic is now cleaner and only runs when the "Run Analysis" button is clicked.
