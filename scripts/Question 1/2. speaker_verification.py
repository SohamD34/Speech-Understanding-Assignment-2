import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from transformers import AutoModel, AutoProcessor, Wav2Vec2Processor, AutoModelForPreTraining,  WavLMModel, WavLMConfig
from peft import LoraConfig, get_peft_model
from utils import *
import csv
import random
import requests
import warnings
warnings.filterwarnings('ignore')
os.chdir('../..')

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


VOXCELEB1_PATH = "data/voxceleb1/"
VOXCELEB2_PATH = "data/voxceleb2/"
TRIAL_PAIRS_PATH = "data/voxceleb1_trial_pairs.csv"
MODEL_CHECKPOINT = "wavlm/wavlm_large_finetune.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")



def load_wavlm_large_from_checkpoint(checkpoint_path):

    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print("Checkpoint loaded successfully")
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Found model state dict in checkpoint under 'model' key")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Found model state dict in checkpoint under 'state_dict' key")
    else:
        state_dict = checkpoint
        print("Assuming checkpoint is directly the state dict")

    print("Initializing WavLM Large model configuration")
    config = WavLMConfig.from_pretrained("microsoft/wavlm-large")
    
    print("Creating WavLM Large model instance")
    model = WavLMModel(config)
    

    if not any(k.startswith('wavlm.') for k in state_dict.keys()) and any(k.startswith('wavlm.') for k in model.state_dict().keys()):
        print("Adding 'wavlm.' prefix to state dict keys")
        state_dict = {'wavlm.' + k: v for k, v in state_dict.items()}
    
    if any(k.startswith('wavlm.') for k in state_dict.keys()) and not any(k.startswith('wavlm.') for k in model.state_dict().keys()):
        print("Removing 'wavlm.' prefix from state dict keys")
        state_dict = {k.replace('wavlm.', ''): v for k in state_dict.keys()}
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded successfully with strict=True")
    except Exception as e:
        print("Attempting to load with strict=False")
        
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded with strict=False (some weights may not have been loaded)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Moving model to device: {device}")
    model = model.to(device)
    
    model.eval()
    print("Model set to evaluation mode")
    
    return model




def load_trial_pairs(csv_path):

    log_text('logs/question1.txt', "Loading trial pairs from CSV...")
    
    pairs = []
    trial_pairs_data = pd.read_csv(csv_path)

    for index, row in trial_pairs_data.iterrows():

        label = int(row['Column1'])
        audio_path1 = row['Column2']
        audio_path2 = row['Column3']

        if os.path.exists('data/voxceleb1/'+ audio_path1):
            audio_path1 = 'data/voxceleb1/'+ audio_path1
        elif os.path.exists('data/voxceleb2/'+ audio_path1):
            audio_path1 = 'data/voxceleb2/'+ audio_path1
        else:
            audio_path1 = None
        
        if os.path.exists('data/voxceleb1/'+audio_path2):
            audio_path2 = 'data/voxceleb1/'+ audio_path2
        elif os.path.exists('data/voxceleb2/'+audio_path2):
            audio_path2 = 'data/voxceleb2/'+ audio_path2
        else:
            audio_path2 = None

        if audio_path1 and audio_path2:
            pairs.append((label, audio_path1, audio_path2))
  
    log_text('logs/question1.txt', f"Loaded {len(pairs)} trial pairs.")
    return pairs



class SpeakerVerificationDataset(Dataset):
    """
    Custom PyTorch Dataset for speaker verification.
    """
    def __init__(self, trial_pairs, voxceleb_path):

        self.trial_pairs = trial_pairs
        self.voxceleb_path = voxceleb_path

    def __len__(self):
        return len(self.trial_pairs)

    def __getitem__(self, idx):
        label, audio_path1, audio_path2 = self.trial_pairs[idx]
        # audio1 = librosa.load(audio_path1)
        # audio2 = librosa.load(audio_path2)
        return audio_path1, audio_path2, label




def evaluate_verification(model, pair_loader, device):

    model.eval()
    similarities = []
    actual_labels = []  # Correctly store all labels
    eer, tar_at_far = 0, 0
    
    with torch.no_grad():
        for batch_idx, (audiopaths1, audiopaths2, labels) in enumerate(pair_loader):   
            
            if batch_idx%50 == 0:
                log_text('logs/question1.txt',f"Processing batch {batch_idx}/{len(pair_loader)}")

            for audio_idx in range(len(audiopaths1)):
                audiopath1 = audiopaths1[audio_idx]
                audiopath2 = audiopaths2[audio_idx]
                label = labels[audio_idx]
                
                audio1, _ = librosa.load(audiopath1, sr=16000)
                audio2, _ = librosa.load(audiopath2, sr=16000)
                
                # Convert to PyTorch tensors and add batch dimension
                audio1 = torch.tensor(audio1, dtype=torch.float32).unsqueeze(0).to(device)
                audio2 = torch.tensor(audio2, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Extract features
                features1 = model(audio1).last_hidden_state.mean(dim=1)
                features2 = model(audio2).last_hidden_state.mean(dim=1)

                # Compute cosine similarity
                similarity = F.cosine_similarity(features1, features2).item()
                
                similarities.append(similarity)
                actual_labels.append(label)  
        
    similarities = np.array(similarities)
    actual_labels = np.array(actual_labels) 
    
    fpr, tpr, _ = roc_curve(actual_labels, similarities)
    diff = np.abs(fpr - (1 - tpr))
    eer_index = np.argmin(diff)
    eer = fpr[eer_index]
    tar_index = np.argmin(np.abs(fpr - 0.01))
    tar_at_far = tpr[tar_index]

    return eer, tar_at_far





if __name__ == "__main__":

    log_text('logs/question1.txt',"Loading feature extractor and model...")
    model = load_wavlm_large_from_checkpoint('scripts/Question 1/models/wavlm_base/wavlm_large_finetune.pth')
    
    log_text('logs/question1.txt',"Creating speaker verification dataset...")
    trial_pairs = load_trial_pairs('data/voxceleb1_trial_pairs.csv')
    speaker_verification_pairs = load_trial_pairs('data/voxceleb1_trial_pairs.csv')
    speaker_verification_dataset = SpeakerVerificationDataset(speaker_verification_pairs, VOXCELEB1_PATH)
    
    log_text('logs/question1.txt',"Creating dataLoader for speaker verification dataset...")
    speaker_verification_loader = DataLoader(speaker_verification_dataset, batch_size=128, shuffle=True)


    log_text('logs/question1.txt',"Evaluating speaker verification...")
    eer, tar_at_far = evaluate_verification(model, speaker_verification_loader, DEVICE)
    log_text('logs/question1.txt',f"\nEER (as %): {eer*100:.2f}")
    log_text('logs/question1.txt',f"TAR at FAR=0.01: {tar_at_far*100:.2f}")
