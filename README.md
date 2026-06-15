# AuraVoice: AI Speech Authenticity Classification Engine

AuraVoice is a machine learning-based acoustic security system engineered to distinguish between **Genuine (Human)** and **Deepfake (AI-synthesized)** speech recordings. By utilizing Mel-Frequency Cepstral Coefficients (MFCCs) and a 2D Convolutional Neural Network (CNN), AuraVoice delivers robust verification scores to secure voice-authenticated applications.

---

## Engine Overview

*   **Acoustic Profiling**: Converts standard speech audio into a normalized 40 x 150 grid of MFCC feature maps.
*   **Convolutional Signal Parsing**: Employs Conv2D layers with pooling and dropout to extract spatial patterns from voice spectrograms.
*   **Pre-trained Weights Integrated**: Ships with `deepfake_audio_model.pth` containing pre-trained weights for instant local validation and inference.
*   **Visual Analysis Dashboard**: A clean, premium dark-themed Streamlit application providing real-time audio playback, waveform graphics, and security score meters.
*   **Inference Command Line**: Quick-verification script to parse vocal files and check integrity metrics directly from the terminal.

---

## Project Structure

The project files are organized at the root directory to ensure compatibility with standard execution environments:

```
├── app.py                   # Premium dark-themed Streamlit web interface
├── predict.py               # Rebranded command-line verification script
├── train_pipeline.py        # Customized model training pipeline code
├── notebook.ipynb           # Optimized Jupyter notebook (caching features for 100x speed)
├── deepfake_audio_model.pth # The pre-trained neural network weights file
├── requirements.txt         # Package dependencies list
├── packages.txt             # Debian system dependencies list
└── README.md                # Project documentation guide
```

---

## Methodology and Classifier Architecture

The system processes signals through the following pipeline:
1. **Raw Speech Audio**: Audio files loaded in WAV, MP3, or FLAC.
2. **16000 Hz Resampling**: Audio is resampled to a clean 16 kHz mono signal.
3. **MFCC Feature Extraction**: 40 Mel-frequency cepstral coefficients are extracted.
4. **Temporal Padding/Truncating**: Audio frames are padded/sliced to a fixed width of 150 frames.
5. **Input Tensor Grid**: Feeds a 1x40x150 feature map into the 2D CNN classifier.
6. **2D CNN Feature Extraction**: Convolutions map channels 1 -> 16 -> 32 -> 64 with max pooling.
7. **Classification Head**: Flattened features route through a dense layer (128 units, 50% dropout) to a Sigmoid activation.


### 1. Acoustic Preprocessing
*   **Resampling**: Standardizes all audio signals to **16 kHz** sample rate.
*   **Mel-Cepstrum Conversion**: Extracts **40 MFCC bands** across the signal length, capturing the vocal tract resonance characteristics.
*   **Temporal Scaling**: Pads short clips or slices longer recordings to a constant length of **150 frames** (~3.0 seconds), generating an input tensor dimensions of `1 x 40 x 150`.

### 2. Network Layout (`AudioCNN`)
*   **Feature Abstraction**: Three successive Convolutional 2D layers extract temporal-frequency features mapping channels `1 -> 16 -> 32 -> 64`.
*   **Dimension Reduction**: 2D Max Pooling follows each activation to downsample features.
*   **Dense Layers**: Flattened vector is processed via a linear layer (`5760 -> 128`) with `Dropout(0.5)` to counter overfitting.
*   **Output Node**: A single sigmoidal neuron yields an authenticity likelihood probability (0 to 1, threshold of 0.5).

---

## Installation & Launch Guide

### System Prerequisites
*   Python 3.9+
*   FFmpeg (required for parsing raw compressed MP3/FLAC formats)

### 1. Install Dependencies
Run the package installation:
```bash
pip install -r requirements.txt
```

### 2. Run the Visual Dashboard
To start the dashboard locally:
```bash
streamlit run app.py
```
*   Browse to the local URL (usually `http://localhost:8501`).
*   Upload any speech clip in `.wav`, `.mp3`, `.flac`, or `.ogg` format.
*   Observe the **Signal Waveform** plot and click **Analyze Authenticity** to inspect prediction metrics.

### 3. Run Command-Line Inference
To verify a speech recording quickly using the terminal:
```bash
python predict.py path/to/vocal_clip.wav
```

### 4. Retraining (Kaggle Ready)
To run full training or cross-dataset validation:
1.  Open a new Notebook on Kaggle.
2.  Upload the **`notebook.ipynb`** file.
3.  Add the **"The Fake-or-Real Dataset"** to the notebook input directories.
4.  Configure the **GPU accelerator** in session settings and click **Run All**.
5.  Download the newly trained `deepfake_audio_model.pth` file.

---

## Verification Metrics & Targets

To pass authenticity certification, predictions on evaluation sets are monitored against the following parameters:

| Parameter | Milestone Threshold | Target Metric |
| :--- | :--- | :--- |
| **Overall Accuracy** | **≥ 80%** | Average correct predictions across all classes. |
| **Equal Error Rate (EER)** | **≤ 12%** | Balance target where False Positive equals False Negative rate. |
| **F1 Score** | **≥ 80%** | Harmonic mean of accuracy metrics. |
| **Per-Class Accuracy** | **≥ 75%** | Individual accuracy rates for genuine and fake recordings. |
| **Confusion Matrix** | **Required** | Tracking detailed prediction errors. |
