#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Author: David Hason Rudd
Description: Unified Feature Extraction for DDE-SER Project.
Extracts Branch A (VMD) and Branch B (HPSS) features from raw audio.
"""

import os
import glob
import numpy as np
import librosa
from vmdpy import VMD
from tqdm import tqdm

# Optimal Parameters
RAV_K = 3
RAV_ALPHA = 1000

def extract_multi_imf_tensor(audio_path, target_shape=(128, 128), max_duration=4.0):
    """Extracts VMD features using optimized parameters."""
    sr_target = 16000
    f, sr = librosa.load(audio_path, sr=sr_target)
    
    f, _ = librosa.effects.trim(f, top_db=30)
    max_samples = int(sr_target * max_duration)
    if len(f) > max_samples:
        f = f[:max_samples]
        
    if len(f) % 2 != 0:
        f = f[:-1]
        
    u, u_hat, omega = VMD(f, RAV_ALPHA, tau=0, K=RAV_K, DC=0, init=1, tol=1e-7)
    
    energies = np.sum(u**2, axis=1)
    top_3_indices = np.argsort(energies)[::-1][:min(3, len(energies))] 
    
    channels = []
    for idx in top_3_indices:
        S = librosa.feature.melspectrogram(y=u[idx], sr=sr_target, n_mels=target_shape[0], fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        if S_dB.shape[1] < target_shape[1]:
            pad_width = target_shape[1] - S_dB.shape[1]
            S_dB = np.pad(S_dB, pad_width=((0,0), (0, pad_width)), mode='constant')
        else:
            S_dB = S_dB[:, :target_shape[1]]
            
        channels.append(S_dB)
        
    while len(channels) < 3:
        channels.append(channels[-1])
        
    return np.dstack(channels)

def extract_hp_tensor(audio_path, target_shape=(128, 128), max_duration=4.0):
    """Extracts Harmonic and Percussive features[cite: 13]."""
    sr_target = 16000
    f, sr = librosa.load(audio_path, sr=sr_target)
    
    f, _ = librosa.effects.trim(f, top_db=30)
    max_samples = int(sr_target * max_duration)
    if len(f) > max_samples:
        f = f[:max_samples]
        
    stft = librosa.stft(f, n_fft=2048, hop_length=512)
    H, P = librosa.decompose.hpss(stft, margin=1.2)
    
    mel_H = librosa.feature.melspectrogram(S=np.abs(H)**2, sr=sr_target, n_mels=target_shape[0], fmax=8000)
    mel_P = librosa.feature.melspectrogram(S=np.abs(P)**2, sr=sr_target, n_mels=target_shape[0], fmax=8000)
    
    mel_H_db = librosa.power_to_db(mel_H, ref=np.max)
    mel_P_db = librosa.power_to_db(mel_P, ref=np.max)
    mel_Hybrid_db = (mel_H_db + mel_P_db) / 2.0
    
    channels = []
    for S_dB in [mel_H_db, mel_P_db, mel_Hybrid_db]:
        if S_dB.shape[1] < target_shape[1]:
            pad_width = target_shape[1] - S_dB.shape[1]
            S_dB = np.pad(S_dB, pad_width=((0,0), (0, pad_width)), mode='constant')
        else:
            S_dB = S_dB[:, :target_shape[1]]
        channels.append(S_dB)
        
    return np.dstack(channels)

def process_unified_dataset(dataset_path, output_dir, dataset_name="ravdess"):
    """Unified Extraction with error catching to ensure 1:1 array lengths[cite: 13]."""
    X_A, X_B, Y, speaker_ids = [], [], [], []
    print(f"\\n[INFO] UNIFIED Extraction for {dataset_name.upper()}")
    
    all_wav_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                all_wav_files.append(os.path.join(root, file))
                
    if not all_wav_files:
        print(f"[ERROR] No .wav files found in '{dataset_path}'")
        return
        
    for file_path in tqdm(sorted(all_wav_files), desc=f"Processing {dataset_name}"):
        file = os.path.basename(file_path)
        try:
            parts = file.split("-")
            emotion_idx = int(parts[2]) - 1 
            speaker_string = parts[6].split(".")[0] 
            speaker_id = int(speaker_string.split("_")[0]) 
            
            tensor_A = extract_multi_imf_tensor(file_path) 
            tensor_B = extract_hp_tensor(file_path)        
            
            X_A.append(tensor_A)
            X_B.append(tensor_B)
            Y.append(emotion_idx)
            speaker_ids.append(speaker_id)
            
        except Exception as e:
            pass 
            
    X_A = np.array(X_A)
    X_B = np.array(X_B)
    Y = np.array(Y, dtype=int)
    speaker_ids = np.array(speaker_ids, dtype=int)
    
    num_classes = len(np.unique(Y))
    Y_one_hot = np.eye(num_classes)[Y]
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, f'{dataset_name}_vmd_128.npy'), X_A)
    np.save(os.path.join(output_dir, f'{dataset_name}_hp_128.npy'), X_B)
    np.save(os.path.join(output_dir, f'{dataset_name}_labels.npy'), Y_one_hot)
    np.save(os.path.join(output_dir, f'{dataset_name}_speakers.npy'), speaker_ids)
    
    print(f"\\n[SUCCESS] Extracted! X_A Shape: {X_A.shape} | X_B Shape: {X_B.shape}")

if __name__ == "__main__":
    OUTPUT_DIR = "data/precomputed_features"
    RAVDESS_PATH = "data/raw_audio/ravdess"
    process_unified_dataset(dataset_path=RAVDESS_PATH, output_dir=OUTPUT_DIR, dataset_name="ravdess")

