
import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
import pathlib

# def load_css(css_path):
#     # css_path can be a str or pathlib.Path
#     path = str(css_path)
#     with open(path) as f:
#         st.html(f"<style>{f.read()}</style>")

# css_path = pathlib.Path(__file__).parent / "styles.css"
# load_css(css_path)

st.header("Welcome to AUTHENTISIGHT!")
st.markdown("---")
st.write("Your essential web tool for verifying the authenticity of digital images. Using advanced computer vision and deep learning algorithms, it swiftly analyzes uploaded photos to determine the likelihood of them being AI-generated (synthetic media) or digitally morphed/altered (deepfakes)")
uploaded_file=st.file_uploader("Upload an Image for Analysis", type=["jpg", "jpeg", "png"])
st.sidebar.title("About")
st.sidebar.write("Home")
st.sidebar.write("Contact")
st.sidebar.header("Configuration")
ela_quality = st.sidebar.slider("ELA Compression Quality:(Default is 90)", 70, 100, 90)

analysis_mode = st.sidebar.radio("Select Analysis Type:",
    ['Deepfake Detection', 'Traditional Forgery (Splicing/Copy-Move)'])
st.sidebar.info("Analysis Model: Loaded and ready for processing.")
if uploaded_file is not None:
    # Display the image once it's uploaded
    st.image(uploaded_file, caption='Image to Analyze', use_container_width=True)

    # --- 3. The Core Logic Trigger ---
    if st.button('Run Traditional Forgery Analysis (ELA)'):
        with st.spinner('Processing... Generating Error Level Analysis (ELA) Image.'):
            pass  # Placeholder for ELA processing code

        
        with st.spinner('Analyzing... This may take a moment to process the image forensics.'):
            # --- REPLACE WITH YOUR MODEL/ANALYSIS CODE ---
            pass

def perform_ela(uploaded_file, quality=90):
    # Open the uploaded image
    original_image = Image.open(uploaded_file).convert('RGB')
    
    # Save the image at the specified quality
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

if uploaded_file is not None:
    st.subheader("Ela Visualization (Bright areas are potential tampered regions)")
    result_ela_image = perform_ela(uploaded_file, quality=ela_quality)
    ela_array = np.array(result_ela_image.convert('L'))
    std_dev = np.std(ela_array)
    TAMPER_THRESHOLD = 20  # Example threshold, adjust based on testing
    st.image(result_ela_image, caption='Error level analysis Map', use_container_width=True)
    
    st.success("✅ Analysis Complete!")

    st.metric(label="Tampering Score (Standard Deviation of ELA)", value=f"{std_dev:.2f}")
    
    if std_dev > TAMPER_THRESHOLD:
        st.subheader("Final Verdict:")
        st.error("⚠️ Warning: The image shows signs of potential tampering or manipulation.")
    else:
        st.subheader("Final Verdict:")
        st.success("✅ The image appears to be authentic with no significant signs of tampering.")
    st.write("Error Level Analysis and noise analysis indicate uniform compression and noise patterns across the image.")
    
            