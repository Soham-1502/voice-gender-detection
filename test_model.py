# test_model.py
import argparse
import pickle
import numpy as np
import librosa
import matplotlib.pyplot as plt
from features_extraction import extract_features

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_gender(audio_path, visualize=False, transcribe=False):
    print(f"\n📂 Loading: {audio_path}")
    
    # Extract features
    features = extract_features(audio_path)
    if features is None:
        print("❌ Feature extraction failed.")
        return

    # Scale features
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probas = model.predict_proba(features_scaled)[0]
    
    gender = "Male" if prediction == 0 else "Female"
    confidence = probas[prediction] * 100
    
    print(f"🎯 Predicted Gender: {gender}")
    print(f"📊 Confidence: {confidence:.2f}%")

    # Optional: Pitch visualization
    if visualize:
        y, sr = librosa.load(audio_path, sr=None)
        pitch = librosa.yin(y, fmin=50, fmax=300)
        plt.plot(pitch)
        plt.title("Pitch Contour")
        plt.xlabel("Frames")
        plt.ylabel("Pitch (Hz)")
        plt.grid(True)
        plt.show()

    # Optional: Transcription
    if transcribe:
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                print(f"🗣️ Transcription:\n{text}")
        except Exception as e:
            print(f"⚠️ Transcription failed: {e}")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test voice gender model on a .wav file")
    parser.add_argument("audio_path", type=str, help="Path to a .wav file")
    parser.add_argument("--visualize", action="store_true", help="Plot pitch graph")
    parser.add_argument("--transcribe", action="store_true", help="Transcribe audio (requires internet)")

    args = parser.parse_args()
    predict_gender(args.audio_path, args.visualize, args.transcribe)
