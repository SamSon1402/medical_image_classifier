import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from models.model import MammogramClassifier
from src.preprocessing import preprocess_mammogram, segment_breast_region
from src.feature_extraction import wavelet_transform, extract_lbp_features

# Page configuration
st.set_page_config(
    page_title="MediScan - Advanced Breast Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a futuristic look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #4e9af1;
    }
    .stButton button {
        background-color: #4e9af1;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #2a7de1;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .analysis-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4e9af1;
    }
    .metric-label {
        font-size: 14px;
        color: #a3a8b8;
    }
    .stProgress .st-bo {
        background-color: #4e9af1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'wavelet_vis' not in st.session_state:
    st.session_state.wavelet_vis = None

# Load the model
@st.cache_resource
def load_model():
    model = MammogramClassifier(num_classes=2)
    try:
        model.load_state_dict(torch.load('models/saved_models/best_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except:
        st.warning("No trained model found. Running in demo mode with simulated predictions.")
        return None

model = load_model()

# Header section
st.title("MediScan: Advanced Breast Cancer Detection")
st.markdown("##### Using Microwave Imaging Technology and Deep Learning")

# Main layout with columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Image Upload & Processing")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a mammogram image", type=["jpg", "jpeg", "png", "bmp"])
    
    # Demo option
    use_demo = st.checkbox("Use a demo image instead")
    
    if use_demo:
        # This would load a demo image
        try:
            demo_image = cv2.imread('data/raw/sample_mammogram.jpg', 0)
            if demo_image is not None:
                st.session_state.original_image = demo_image
            else:
                # Fallback to generating a dummy image if file not found
                st.session_state.original_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
                st.info("Using a randomly generated image for demonstration")
        except:
            # Generate a dummy image
            st.session_state.original_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            st.info("Using a randomly generated image for demonstration")
    
    elif uploaded_file is not None:
        # Load and process the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.session_state.original_image = image
    
    # Display original image if available
    if 'original_image' in st.session_state:
        st.image(st.session_state.original_image, caption="Original Image", width=300)
        
        # Process button
        if st.button("Process Image"):
            with st.spinner("Processing image..."):
                # Preprocess the image
                processed = preprocess_mammogram(st.session_state.original_image)
                segmented = segment_breast_region(processed)
                st.session_state.processed_image = segmented
                
                # Extract features for visualization
                wavelet_features = wavelet_transform(segmented)
                lbp_features = extract_lbp_features(segmented)
                st.session_state.features = {
                    'wavelet': wavelet_features,
                    'lbp': lbp_features
                }
                
                # Generate wavelet decomposition for visualization
                coeffs = pywt.wavedec2(segmented, 'db1', level=2)
                # Arrange coefficients for visualization
                approx, (h1, v1, d1), (h2, v2, d2) = coeffs
                # Normalize for visualization
                approx_norm = (approx - approx.min()) / (approx.max() - approx.min())
                h1_norm = (h1 - h1.min()) / (h1.max() - h1.min() + 1e-10)
                v1_norm = (v1 - v1.min()) / (v1.max() - v1.min() + 1e-10)
                d1_norm = (d1 - d1.min()) / (d1.max() - d1.min() + 1e-10)
                
                # Create the visualization
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                axes[0, 0].imshow(approx_norm, cmap='viridis')
                axes[0, 0].set_title('Approximation')
                axes[0, 1].imshow(h1_norm, cmap='viridis')
                axes[0, 1].set_title('Horizontal Detail')
                axes[1, 0].imshow(v1_norm, cmap='viridis')
                axes[1, 0].set_title('Vertical Detail')
                axes[1, 1].imshow(d1_norm, cmap='viridis')
                axes[1, 1].set_title('Diagonal Detail')
                st.session_state.wavelet_vis = fig
                
                # Make prediction
                if model is not None:
                    # Prepare image for the model
                    img_tensor = torch.tensor(segmented).float().unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        output = model(img_tensor)
                        probabilities = F.softmax(output, dim=1)
                    st.session_state.prediction = probabilities[0].numpy()
                else:
                    # Simulate prediction for demo
                    st.session_state.prediction = np.array([0.82, 0.18])  # Benign, Malignant

with col2:
    st.markdown("### Analysis Results")
    
    if st.session_state.processed_image is not None:
        # Display processed image
        st.image(st.session_state.processed_image, caption="Processed & Segmented Image", width=300)
        
        # Display prediction results
        if st.session_state.prediction is not None:
            st.markdown("#### Diagnostic Assessment")
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{st.session_state.prediction[0]*100:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Benign Probability</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_res2:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{st.session_state.prediction[1]*100:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Malignant Probability</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = max(st.session_state.prediction) * 100,
                title = {'text': "Prediction Confidence"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#4e9af1"},
                    'steps': [
                        {'range': [0, 50], 'color': "#EF553B"},
                        {'range': [50, 75], 'color': "#FFA15A"},
                        {'range': [75, 100], 'color': "#00CC96"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': max(st.session_state.prediction) * 100
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor="#1e2130",
                font=dict(color="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification result
            result = "Benign" if st.session_state.prediction[0] > st.session_state.prediction[1] else "Malignant"
            result_color = "#00CC96" if result == "Benign" else "#EF553B"
            
            st.markdown(f"""
            <div style="background-color: #1e2130; padding: 20px; border-radius: 10px; margin-top: 20px; text-align: center;">
                <h3 style="margin-bottom: 10px;">Classification Result</h3>
                <div style="font-size: 28px; font-weight: bold; color: {result_color};">{result}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display feature visualizations
        if st.session_state.features is not None:
            with st.expander("Feature Analysis", expanded=False):
                if st.session_state.wavelet_vis is not None:
                    st.pyplot(st.session_state.wavelet_vis)
                    
                # Plot LBP features
                lbp_fig = px.bar(
                    x=np.arange(len(st.session_state.features['lbp'])),
                    y=st.session_state.features['lbp'],
                    labels={'x': 'LBP Bins', 'y': 'Frequency'},
                    title='Local Binary Pattern Features'
                )
                lbp_fig.update_layout(
                    paper_bgcolor="#1e2130",
                    plot_bgcolor="#1e2130",
                    font=dict(color="white")
                )
                st.plotly_chart(lbp_fig, use_container_width=True)

# Explanation section
with st.expander("How it works", expanded=False):
    st.markdown("""
    ### Microwave Imaging Technology

    This application demonstrates an AI-powered approach to breast cancer detection using microwave imaging technology. The process involves:

    1. **Image Preprocessing**: The uploaded mammogram is preprocessed to enhance features and reduce noise.
    
    2. **Feature Extraction**: 
       - Wavelet transforms are applied to extract frequency-based features
       - Local Binary Patterns (LBP) capture texture information
       - These techniques are particularly relevant to microwave imaging analysis
    
    3. **Deep Learning Classification**: A convolutional neural network analyzes the extracted features to classify the image.
    
    4. **Visualization**: Results are presented with confidence metrics and feature visualizations for interpretability.

    This approach mimics the signal processing techniques used in actual microwave imaging systems for breast cancer detection.
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; color: #a3a8b8;">
    <p>MediScan - Advanced Breast Cancer Detection System</p>
    <p style="font-size: 12px;">This is a demonstration prototype. Not for clinical use.</p>
</div>
""", unsafe_allow_html=True)