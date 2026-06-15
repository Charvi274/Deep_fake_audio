import torch
import librosa
import numpy as np
import argparse
import sys
from train_pipeline import AudioCNN, extract_features

def predict_audio(file_path, model_path='deepfake_audio_model.pth'):
    """
    Determine if a given audio sample is Genuine (Human-voiced) or a Deepfake (AI-synthesized).
    """
    # CPU/GPU Device routing
    computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the neural model
    vocal_classifier = AudioCNN()
    try:
        # Load weights
        vocal_classifier.load_state_dict(torch.load(model_path, map_location=computation_device))
        vocal_classifier.to(computation_device)
        vocal_classifier.eval()
    except Exception as err:
        print(f"Error: Unable to load model weights: {err}")
        print("Please check that the weights file exists and matches the architecture dimensions.")
        sys.exit(1)
        
    # Extract MFCC audio features
    vocal_features = extract_features(file_path)
    if vocal_features is None:
        print("Error: Feature extraction failed on the target audio clip.")
        sys.exit(1)
        
    # Reshape matching (Batch=1, Channel=1, N_MFCC=40, MAX_LEN=150)
    input_features = vocal_features[np.newaxis, np.newaxis, ...]
    torch_tensor_inputs = torch.tensor(input_features, dtype=torch.float32).to(computation_device)
    
    # Run prediction forwards
    with torch.no_grad():
        model_output_probability = vocal_classifier(torch_tensor_inputs).item()
        
    # Classification: Genuine = 1, Deepfake = 0
    is_genuine_human = model_output_probability >= 0.5
    confidence_factor = model_output_probability if is_genuine_human else (1.0 - model_output_probability)
    label_classification = "Genuine (Human Speech)" if is_genuine_human else "Deepfake (AI-Generated)"
    
    print("\n----------------------------------------")
    print("      AURAVOICE SPEECH VERIFICATION     ")
    print("----------------------------------------")
    print(f"Target:      {file_path}")
    print(f"Prediction:  {label_classification}")
    print(f"Confidence:  {confidence_factor * 100:.2f}%")
    print(f"Probability: {model_output_probability:.4f}")
    print("----------------------------------------\n")
    
    return label_classification, confidence_factor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speech Authenticity Verification Tool")
    parser.add_argument('audio_path', type=str, help='Path to speech audio sample (.wav, .mp3, .flac)')
    parser.add_argument('--model', type=str, default='deepfake_audio_model.pth', help='Path to trained PyTorch weights')
    
    cli_arguments = parser.parse_args()
    predict_audio(cli_arguments.audio_path, cli_arguments.model)
