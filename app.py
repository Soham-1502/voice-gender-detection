import streamlit as st
import numpy as np
import pickle
import soundfile as sf
import librosa
import tempfile
import sounddevice as sd
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from features_extraction import extract_features
import whisper

# ----------------- App Config -----------------
st.set_page_config(
    page_title="🎤 Voice Gender Detector", 
    page_icon="🎧", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Dark Theme CSS with Colorful Gradients
st.markdown("""
    <style>
        /* Main app background with gradient */
        .stApp {
            background: linear-gradient(135deg, #0c1015 0%, #1a1d29 25%, #2d1b3d 50%, #1e2139 75%, #0f172a 100%);
            background-attachment: fixed;
        }
        
        /* Enhanced Sidebar styling with navy background */
        .css-1d391kg {
            background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        }
        
        .stSidebar > div:first-child {
            background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        }
        
        /* Main header styling */
        .main-header {
            background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 2rem;
            text-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
        }
        
        /* Section headers */
        h1, h2, h3 {
            color: #f8fafc !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }
        
        /* Regular text */
        p, .stMarkdown, .stText {
            color: #e2e8f0 !important;
        }
        
        /* Fix file uploader text opacity - make filename fully opaque */
        .uploadedFile {
            color: #FFFFFF !important;
            opacity: 1 !important;
        }
        
        /* File uploader filename text fix */
        .stFileUploader label, 
        .stFileUploader div[data-testid="stFileUploaderDropzoneInstructions"],
        .stFileUploader div[data-testid="stFileUploaderDropzone"] p,
        .stFileUploader div[data-testid="stFileUploaderDropzone"] span,
        .stFileUploader .uploadedFileName {
            color: #f8fafc !important;
            opacity: 1 !important;
        }
        
        /* Enhanced buttons with gradient and hover effects */
        .stButton > button {
            background: linear-gradient(45deg, #6366f1, #8b5cf6) !important;
            color: white !important;
            font-weight: bold !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 1.5rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
            margin: 0.5rem 0 !important;
            width: 100% !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(45deg, #7c3aed, #a855f7) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5) !important;
        }
        
        /* Predict button special styling */
        .predict-button > button {
            background: linear-gradient(45deg, #10b981, #059669) !important;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
        }
        
        .predict-button > button:hover {
            background: linear-gradient(45deg, #059669, #047857) !important;
            box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5) !important;
        }
        
        /* Features button styling */
        .features-button > button {
            background: linear-gradient(45deg, #f59e0b, #d97706) !important;
            box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3) !important;
        }
        
        .features-button > button:hover {
            background: linear-gradient(45deg, #d97706, #b45309) !important;
            box-shadow: 0 8px 25px rgba(245, 158, 11, 0.5) !important;
        }
        
        /* Transcribe button styling */
        .transcribe-button > button {
            background: linear-gradient(45deg, #ef4444, #dc2626) !important;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3) !important;
        }
        
        .transcribe-button > button:hover {
            background: linear-gradient(45deg, #dc2626, #b91c1c) !important;
            box-shadow: 0 8px 25px rgba(239, 68, 68, 0.5) !important;
        }
        
        /* File uploader styling */
        .stFileUploader > div > div {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 2px dashed #6366f1 !important;
            border-radius: 12px !important;
            padding: 2rem !important;
        }
        
        /* Input fields */
        .stTextInput > div > div > input, .stSelectbox > div > div > div {
            background: rgba(30, 41, 59, 0.8) !important;
            color: #f8fafc !important;
            border: 2px solid #6366f1 !important;
            border-radius: 8px !important;
        }
        
        /* Metrics styling */
        .metric-container {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(51, 65, 85, 0.9)) !important;
            padding: 1.5rem !important;
            border-radius: 12px !important;
            border: 1px solid rgba(139, 92, 246, 0.3) !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
            margin: 0.5rem 0 !important;
        }
        
        .metric-container .metric-label {
            color: #94a3b8 !important;
            font-size: 0.9rem !important;
        }
        
        .metric-container .metric-value {
            color: #f8fafc !important;
            font-size: 1.8rem !important;
            font-weight: bold !important;
        }
        
        /* Enhanced Sidebar radio buttons with navy theme */
        .stRadio > div {
            background: rgba(30, 41, 59, 0.8) !important;
            padding: 1rem !important;
            border-radius: 8px !important;
        }
        
        .stRadio label {
            color: #f8fafc !important;
        }
        
        /* Sidebar text styling */
        .stSidebar .stMarkdown, .stSidebar p, .stSidebar h1, .stSidebar h2, .stSidebar h3 {
            color: #f8fafc !important;
        }
        
        /* Audio player styling */
        .stAudio {
            margin: 1rem 0 !important;
        }
        
        /* Success/Error messages */
        .stSuccess {
            background: linear-gradient(90deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.2)) !important;
            border: 1px solid #10b981 !important;
            border-radius: 8px !important;
        }
        
        .stError {
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.2)) !important;
            border: 1px solid #ef4444 !important;
            border-radius: 8px !important;
        }
        
        /* Info messages */
        .stInfo {
            background: linear-gradient(90deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2)) !important;
            border: 1px solid #6366f1 !important;
            border-radius: 8px !important;
        }
        
        /* Slider styling */
        .stSlider > div > div > div {
            color: #f8fafc !important;
        }
        
        /* Text area styling */
        .stTextArea > div > div > textarea {
            background: rgba(30, 41, 59, 0.8) !important;
            color: #f8fafc !important;
            border: 2px solid #6366f1 !important;
            border-radius: 8px !important;
        }
        
        /* Spinner customization */
        .stSpinner > div {
            border-top-color: #8b5cf6 !important;
        }
        
        /* Vertical button container */
        .button-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Main header with gradient
st.markdown("<h1 class='main-header'>🎙️ AI Voice Gender Detection Studio</h1>", unsafe_allow_html=True)

# ----------------- Load Models -----------------
@st.cache_resource
def load_transcription_model():
    return whisper.load_model("base")

transcription_model = load_transcription_model()

@st.cache_resource
def load_model_and_scaler():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except Exception as e:
        st.error(f"❌ Model or scaler not found: {e}")
        return None, None

model, scaler = load_model_and_scaler()

# ----------------- Functions -----------------
def transcribe_audio(file_path):
    try:
        result = transcription_model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        st.exception(f"Transcription error: {str(e)}")
        return "Transcription failed."

def predict_gender(file_path):
    if model is None or scaler is None:
        return None, None
    try:
        features = extract_features(file_path)
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0]
        return ("Male" if prediction == 0 else "Female", max(prob) * 100)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def extract_features_for_visualization(file_path):
    y, sr = librosa.load(file_path, sr=None)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rmse = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return {
        "y": y, "sr": sr,
        "pitch": pitch, "mfcc": mfcc,
        "zcr": zcr, "rmse": rmse,
        "spectral_centroids": centroid
    }

def plot_audio_features(features_data):
    time_frames = len(features_data['spectral_centroids'])
    time_axis = np.linspace(0, len(features_data['y']) / features_data['sr'], time_frames)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis, 
        y=features_data['spectral_centroids'], 
        name="Spectral Centroid",
        line=dict(color='#8b5cf6', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=time_axis, 
        y=features_data['zcr'], 
        name="Zero Crossing Rate",
        line=dict(color='#10b981', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=time_axis, 
        y=features_data['rmse'], 
        name="RMS Energy",
        line=dict(color='#f59e0b', width=3)
    ))
    
    fig.update_layout(
        title="🎵 Audio Feature Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Feature Values",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        legend=dict(
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor='rgba(139, 92, 246, 0.3)',
            borderwidth=1
        )
    )
    
    return fig

def plot_mfcc(mfcc):
    fig = px.imshow(
        mfcc, 
        aspect='auto', 
        color_continuous_scale='Viridis', 
        title="🎼 Mel-Frequency Cepstral Coefficients (MFCCs)"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc')
    )
    return fig

# Custom metric display function
def display_metric(label, value, icon="📊"):
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

# ----------------- Enhanced UI with Vertical Navigation -----------------
# Enhanced Sidebar with navy background theme and vertical navigation
with st.sidebar:
    st.markdown("### 🎛️ Navigation Panel")
    menu = st.radio(
        "Choose an option:",
        ["📂 Upload Audio File", "🎤 Live Recording", "ℹ️ Model Information"],
        index=0
    )

# ========== Upload Section ==========
if menu == "📂 Upload Audio File":
    st.markdown("### 📁 Upload Your Audio Sample")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        audio_file = st.file_uploader(
            "Choose a .wav file", 
            type=["wav"],
            help="Upload a .wav audio file for gender prediction"
        )
    
    if audio_file:
        st.success("✅ File uploaded successfully!")
        st.audio(audio_file)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        # Vertical button layout
        st.markdown("### 🔧 Analysis Tools")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="predict-button">', unsafe_allow_html=True)
            predict_btn = st.button("🧠 Predict Gender", use_container_width=True, key="predict1")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="features-button">', unsafe_allow_html=True)
            features_btn = st.button("📈 Show Features", use_container_width=True, key="features1")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="transcribe-button">', unsafe_allow_html=True)
            transcribe_btn = st.button("🎯 Transcribe Audio", use_container_width=True, key="transcribe1")
            st.markdown('</div>', unsafe_allow_html=True)

        # Handle button actions
        if predict_btn:
            with st.spinner("🔮 Analyzing voice patterns..."):
                gender, confidence = predict_gender(tmp_path)
                if gender:
                    st.markdown("### 🎯 Prediction Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        display_metric("Predicted Gender", gender, "👤")
                    with col2:
                        display_metric("Confidence Level", f"{confidence:.1f}%", "📊")

        if features_btn:
            with st.spinner("📊 Extracting audio features..."):
                data = extract_features_for_visualization(tmp_path)
                st.markdown("### 📈 Audio Feature Visualization")
                st.plotly_chart(plot_audio_features(data), use_container_width=True)
                st.plotly_chart(plot_mfcc(data['mfcc']), use_container_width=True)

        if transcribe_btn:
            with st.spinner("🎧 Transcribing audio..."):
                transcript = transcribe_audio(tmp_path)
                st.markdown("### 📝 Audio Transcription")
                st.text_area("Transcribed Text", transcript, height=150, key="transcript1")

# ========== Live Recording Section ==========
elif menu == "🎤 Live Recording":
    st.markdown("### 🎙️ Record Your Voice Live")
    
    col1, col2 = st.columns(2)
    with col1:
        duration = st.slider("⏱️ Recording Duration (seconds)", 1, 10, 3)
    with col2:
        sample_rate = st.selectbox("🎚️ Sample Rate (Hz)", [16000, 22050, 44100], index=1)

    # Recording button
    if st.button("🎤 Start Recording", use_container_width=True, key="record"):
        st.info(f"🎧 Recording for {duration} seconds... Please speak now!")
        
        with st.spinner("Recording in progress..."):
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()
            recording = np.squeeze(recording)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_wav_path = tmp.name
        sf.write(temp_wav_path, recording, sample_rate, subtype='PCM_16')

        st.session_state['recorded_file_path'] = temp_wav_path
        
        # Auto-predict after recording
        with st.spinner("🔮 Analyzing your voice..."):
            gender, confidence = predict_gender(temp_wav_path)
            st.session_state['predicted_gender'] = gender
            st.session_state['confidence'] = confidence

        st.success("✅ Recording completed and analyzed!")
        st.audio(temp_wav_path)

    # Display results and controls for recorded audio
    if 'recorded_file_path' in st.session_state:
        audio_path = st.session_state['recorded_file_path']
        
        if 'predicted_gender' in st.session_state and st.session_state['predicted_gender']:
            st.markdown("### 🎯 Live Recording Results")
            col1, col2 = st.columns(2)
            with col1:
                display_metric("Predicted Gender", st.session_state['predicted_gender'], "👤")
            with col2:
                display_metric("Confidence Level", f"{st.session_state['confidence']:.1f}%", "📊")

        # Analysis tools for recorded audio
        st.markdown("### 🔧 Additional Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="features-button">', unsafe_allow_html=True)
            if st.button("📊 Show Audio Features", use_container_width=True, key="features2"):
                try:
                    with st.spinner("📊 Extracting features..."):
                        data = extract_features_for_visualization(audio_path)
                        st.plotly_chart(plot_audio_features(data), use_container_width=True)
                        st.plotly_chart(plot_mfcc(data['mfcc']), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Feature extraction error: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="transcribe-button">', unsafe_allow_html=True)
            if st.button("📝 Transcribe Recording", use_container_width=True, key="transcribe2"):
                with st.spinner("🎧 Transcribing..."):
                    transcript = transcribe_audio(audio_path)
                    st.markdown("### 📝 Transcription Results")
                    st.text_area("Transcribed Text", transcript, height=150, key="transcript2")
            st.markdown('</div>', unsafe_allow_html=True)

# ========== Model Info Section ==========
elif menu == "ℹ️ Model Information":
    st.markdown("### 🤖 AI Model Details")
    
    if model:
        st.success("✅ Model loaded and ready for predictions!")
        
        # Model specifications in cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="metric-container">
                    <h4>🧠 Model Architecture</h4>
                    <ul>
                        <li><strong>Algorithm:</strong> Support Vector Machine / Random Forest</li>
                        <li><strong>Input Format:</strong> WAV audio files</li>
                        <li><strong>Output Classes:</strong> Male (0), Female (1)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="metric-container">
                    <h4>📊 Feature Extraction</h4>
                    <ul>
                        <li><strong>Pitch:</strong> Fundamental frequency analysis</li>
                        <li><strong>MFCC:</strong> 13 coefficients</li>
                        <li><strong>ZCR:</strong> Zero crossing rate</li>
                        <li><strong>RMS:</strong> Root mean square energy</li>
                        <li><strong>Spectral Centroid:</strong> Frequency distribution</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Model not loaded. Please ensure model files are available.")
        st.code("python train_model.py", language="bash")

    st.markdown("### 🛠️ Training Instructions")
    st.markdown("""
        <div class="metric-container">
            <h4>📋 Setup Process</h4>
            <ol>
                <li><strong>Data Preparation:</strong> Place .wav files in <code>data/male/</code> and <code>data/female/</code> directories</li>
                <li><strong>Model Training:</strong> Run <code>python train_model.py</code> to train the model</li>
                <li><strong>Model Files:</strong> This generates <code>model.pkl</code> and <code>scaler.pkl</code> files</li>
                <li><strong>Deployment:</strong> Launch the app with <code>streamlit run app.py</code></li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #94a3b8; margin-top: 2rem;'>
        <p>🎵 Built with Streamlit • Powered by Machine Learning • Enhanced with AI 🚀</p>
    </div>
""", unsafe_allow_html=True)