import os
import glob
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Audio processing constants (must match reference dimensions for pre-trained weights)
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 150 
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

def extract_features(file_path):
    """
    Extract MFCC acoustic features from speech files.
    """
    try:
        # Load and resample
        vocal_signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc_features = librosa.feature.mfcc(y=vocal_signal, sr=sr, n_mfcc=N_MFCC)
        
        # Temporal shape normalization (Pad or truncate)
        _, frames_count = mfcc_features.shape
        if frames_count < MAX_LEN:
            padding_width = MAX_LEN - frames_count
            mfcc_features = np.pad(mfcc_features, pad_width=((0, 0), (0, padding_width)), mode='constant')
        else:
            mfcc_features = mfcc_features[:, :MAX_LEN]
            
        return mfcc_features
    except Exception as err:
        print(f"Error extracting features from {file_path}: {err}")
        return None

class AudioDataset(Dataset):
    """
    Vocal dataset loading utility for training.
    """
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path = self.data_list[idx]
        label = self.labels[idx]
        features = extract_features(file_path)
        
        if features is None:
            features = np.zeros((N_MFCC, MAX_LEN))
            
        # Add channel dimension: (1, N_MFCC, MAX_LEN)
        features = features[np.newaxis, ...]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class AudioCNN(nn.Module):
    """
    Convolutional Neural Network (CNN) model for voice authenticity analysis.
    This architecture must map exactly to reference dimensions for checkpoint compatibility.
    """
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Calculate intermediate layer flattened dimension
        self.flat_size = 64 * (N_MFCC // 8) * (MAX_LEN // 8)
        
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, self.flat_size)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return self.sigmoid(x).squeeze()

def compute_eer(y_true, y_scores):
    """
    Calculate Equal Error Rate (EER) of classification prediction scores.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    # Index where false positive rate equals false negative rate
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    equal_error_rate = fpr[eer_index]
    return equal_error_rate

def load_data(data_dir):
    """
    Locate speech audio samples recursively based on labels.
    """
    speech_files = []
    class_labels = []
    
    # Subdirectory structure templates
    vocal_genuine_dir = os.path.join(data_dir, 'genuine')
    vocal_spoof_dir = os.path.join(data_dir, 'spoof')
    
    if os.path.exists(vocal_genuine_dir) and os.path.exists(vocal_spoof_dir):
        for f in glob.glob(os.path.join(vocal_genuine_dir, '*.wav')) + glob.glob(os.path.join(vocal_genuine_dir, '*.flac')):
            speech_files.append(f)
            class_labels.append(1) # Genuine Speech = 1
        for f in glob.glob(os.path.join(vocal_spoof_dir, '*.wav')) + glob.glob(os.path.join(vocal_spoof_dir, '*.flac')):
            speech_files.append(f)
            class_labels.append(0) # AI Synthesized Speech = 0
    else:
        print(f"Warning: Expected paths not found. Fallback scanning files in {data_dir}.")
        all_wavs = glob.glob(os.path.join(data_dir, '**', '*.wav'), recursive=True) + \
                   glob.glob(os.path.join(data_dir, '**', '*.flac'), recursive=True)
        for f in all_wavs:
            speech_files.append(f)
            # Match based on file labeling
            if 'spoof' in f.lower() or 'fake' in f.lower():
                class_labels.append(0)
            else:
                class_labels.append(1)
                
    return speech_files, class_labels

def train():
    """
    Execution training routine for retraining.
    """
    target_data_dir = 'dataset/LA_norm/train'
    print(f"Scanning target database: {target_data_dir}")
    
    vocal_files, labels = load_data(target_data_dir)
    if not vocal_files:
        print("Data loading failed. Check that dataset is present.")
        return
        
    print(f"Loaded {len(vocal_files)} samples.")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(vocal_files, labels, test_size=0.2, random_state=42, stratify=labels)
    
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    
    classifier_model = AudioCNN().to(device)
    loss_criterion = nn.BCELoss()
    model_optimizer = optim.Adam(classifier_model.parameters(), lr=LEARNING_RATE)
    
    best_f1_value = 0.0
    
    for epoch in range(EPOCHS):
        classifier_model.train()
        epoch_cumulative_loss = 0.0
        
        for inputs, targets in tqdm(train_dataloader, desc=f"Iteration {epoch+1}/{EPOCHS}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            model_optimizer.zero_grad()
            outputs = classifier_model(inputs)
            loss = loss_criterion(outputs, targets)
            loss.backward()
            model_optimizer.step()
            
            epoch_cumulative_loss += loss.item()
            
        print(f"Epoch {epoch+1} Complete | Average Loss: {epoch_cumulative_loss/len(train_dataloader):.4f}")
        
        # Test Validation loop
        classifier_model.eval()
        validation_targets = []
        validation_outputs = []
        
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = classifier_model(inputs)
                
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                    
                validation_targets.extend(targets.cpu().numpy())
                validation_outputs.extend(outputs.cpu().numpy())
                
        validation_targets = np.array(validation_targets)
        validation_outputs = np.array(validation_outputs)
        binary_predictions = (validation_outputs > 0.5).astype(int)
        
        accuracy_val = accuracy_score(validation_targets, binary_predictions)
        f1_val = f1_score(validation_targets, binary_predictions)
        eer_val = compute_eer(validation_targets, validation_outputs)
        
        print(f"Metrics: Accuracy: {accuracy_val:.4f} | F1: {f1_val:.4f} | EER: {eer_val:.4f}")
        
        if f1_val > best_f1_value:
            best_f1_value = f1_val
            torch.save(classifier_model.state_dict(), 'deepfake_audio_model.pth')
            print("Weights file updated.")
            
    # Final performance output
    print("\n================ FINAL SYSTEM EVALUATION ================")
    classifier_model.load_state_dict(torch.load('deepfake_audio_model.pth'))
    classifier_model.eval()
    
    final_targets = []
    final_outputs = []
    
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = classifier_model(inputs)
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            final_targets.extend(targets.cpu().numpy())
            final_outputs.extend(outputs.cpu().numpy())
            
    final_preds = (np.array(final_outputs) > 0.5).astype(int)
    
    print("Confusion Matrix:")
    print(confusion_matrix(final_targets, final_preds))
    print(f"Accuracy: {accuracy_score(final_targets, final_preds):.4f}")
    print(f"F1 Score: {f1_score(final_targets, final_preds):.4f}")
    print(f"Equal Error Rate (EER): {compute_eer(final_targets, final_outputs):.4f}")
    print("=========================================================\n")

if __name__ == '__main__':
    train()
