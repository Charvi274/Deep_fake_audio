import streamlit as st
import tempfile
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
from predict import predict_audio

# Page Configuration
st.set_page_config(
    page_title="AuraVoice - AI Speech Authenticator", 
    page_icon=None, 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom styling for high contrast light theme with premium colors
st.markdown("""
    <style>
    /* Global Background and Typography */
    .stApp {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Section Container */
    .banner-container {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #ECFDF5 0%, #EFF6FF 100%);
        border-radius: 12px;
        border: 1px solid #C0F2FE;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.05);
    }
    
    .banner-title {
        background: linear-gradient(90deg, #0891B2 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        display: inline-block;
    }
    
    .banner-subtitle {
        color: #4B5563;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Upload Drag-and-Drop Area */
    .stFileUploader section {
        background-color: #F9FAFB !important;
        border: 2px dashed #0891B2 !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        color: #1F2937 !important;
    }
    
    /* Custom Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #0891B2 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 10px rgba(124, 58, 237, 0.2) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 15px rgba(124, 58, 237, 0.4) !important;
    }
    
    /* Styled Metric Cards */
    div[data-testid="stMetricContainer"] {
        background-color: #F3F4F6;
        border: 1px solid #E5E7EB;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        color: #1F2937 !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #4B5563 !important;
    }
    
    /* Custom Result Cards */
    .status-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1.5rem;
        text-align: center;
        border: 1px solid;
    }
    
    .verified-pass {
        background-color: #D1FAE5;
        border-color: #10B981;
        color: #065F46;
    }
    
    .verified-fail {
        background-color: #FEE2E2;
        border-color: #EF4444;
        color: #991B1B;
    }
    
    /* Base labels styling */
    p, span, label, h3, h4 {
        color: #1F2937 !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Main Banner Layout
st.markdown("""
    <div class="banner-container">
        <h1 class="banner-title">AuraVoice</h1>
        <p class="banner-subtitle">Acoustic Authentication & Deepfake Detection Engine</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("Assess the authenticity of speech samples. Upload an audio recording to identify if the sound profile matches biological human vocalization or synthetic AI modeling.")

# Custom Gradient Divider
st.markdown("<div style='height: 2px; background: linear-gradient(90deg, #0891B2, #7C3AED); margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

# File Drag-and-Drop Area
st.markdown("### Speech Sample Upload")
audio_file = st.file_uploader(
    "Upload wav, mp3, flac, or ogg file",
    type=['wav', 'flac', 'mp3', 'ogg'],
    label_visibility="collapsed"
)

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    
    # Save uploaded file contents into a temporary WAV format for feature parsing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        temp_audio.write(audio_file.getvalue())
        temp_audio_path = temp_audio.name
        
    # Generate visual waveform of the speech signal
    with st.spinner("Analyzing signal structure..."):
        try:
            y, sr = librosa.load(temp_audio_path, sr=16000)
            
            # Matplotlib plotting (Styled for clean light mode contrast)
            fig, ax = plt.subplots(figsize=(8, 2.5))
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#F9FAFB')
            librosa.display.waveshow(y, sr=sr, color='#0891B2', alpha=0.8, ax=ax)
            ax.set_xlabel("Time (s)", color='#4B5563')
            ax.set_ylabel("Amplitude", color='#4B5563')
            ax.tick_params(colors='#4B5563')
            ax.spines['bottom'].set_color('#E5E7EB')
            ax.spines['left'].set_color('#E5E7EB')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            
            st.markdown("#### Signal Waveform Representation")
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Could not render signal plot: {e}")
            
    # Trigger Analysis Columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        check_authenticity = st.button("Analyze Authenticity", use_container_width=True)
        
    if check_authenticity:
        with st.spinner("Decoding vocal properties..."):
            model_weights = 'deepfake_audio_model.pth'
            if not os.path.exists(model_weights):
                st.error("Model weights file 'deepfake_audio_model.pth' is missing. Please ensure it is present in the workspace.")
            else:
                try:
                    prediction, confidence = predict_audio(temp_audio_path, model_weights)
                    
                    st.markdown("### Detection Analysis Report")
                    
                    # Columns to show results
                    metric_col1, metric_col2 = st.columns(2)
                    
                    # Class parsing
                    is_human = "Genuine" in prediction
                    
                    metric_col1.metric("Determination", "Human Voice" if is_human else "AI Synthetic")
                    metric_col2.metric("Confidence Score", f"{confidence * 100:.2f}%")
                    
                    # Banner status card
                    if is_human:
                        st.markdown("""
                            <div class="status-card verified-pass">
                                <h3>Human Authentication Verified</h3>
                                <p>The vocal profiles, pitch variance, and harmonic resonances match authentic biological speech patterns.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="status-card verified-fail">
                                <h3>Synthetic Signal Detected</h3>
                                <p>Warning: Spectral matching indicates synthetic audio. The signal matches structures produced by deepfake speech generators.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Inference execution failed: {e}")
                finally:
                    # Clean up file
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
else:
    st.info("**Ready**: Upload a speech recording above. The dashboard will analyze its waveform structure and run neural classification to detect deepfake properties.")
