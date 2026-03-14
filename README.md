# dde-speech-emotion-recognition
# DDE-SER: Dual-Decomposition Ensemble Framework for Speech Emotion Recognition

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Official implementation** of the paper: *"DDE-SER: A Dual-Decomposition Ensemble Framework Fusing Adaptive Variational Modes and Harmonic-Percussive Spectrograms for Speech Emotion Recognition"* by David Hason Rudd et al.

---

## 📖 Overview

Speech Emotion Recognition (SER) is persistently challenged by the non-stationary nature of vocal affect and transient, broadband noise. Standard deep learning pipelines often process spectrograms as monolithic inputs, failing to isolate orthogonal acoustic cues (e.g., separating tonal pitch from explosive consonants). 

**DDE-SER** is a novel, dual-branch deep learning framework that bridges 1D adaptive frequency decomposition with 2D structural time-frequency mapping. By utilizing **VGG-optiVMD** at the raw signal level and **Harmonic-Percussive (HP) decomposition** at the spectrogram level, the architecture successfully disentangles high-arousal emotional states, mitigating class confusion without requiring massive, computationally heavy transformer embeddings.

## 🧠 Architecture Methodology

The framework processes raw speech signals through a comprehensive, parallel pipeline:

1. **Branch A (1D Adaptive Mode Extraction):** Employs **VGG-optiVMD**, an adaptive, non-recursive filter bank that decomposes the waveform into Intrinsic Mode Functions (IMFs). The optimal hyperparameters ($K$ modes and penalty $\alpha$) are dynamically selected using network loss feedback to minimize mode-mixing and capture narrow-band emotional cues.
2. **Branch B (2D Structural Spectrogram Extraction):** Maps the signal to the time-frequency domain using STFT and applies orthogonal median filtering. This structurally separates continuous tonal formants (Harmonics) from broadband transient noises (Percussives), which are then mapped to the Mel scale to form a 2D hybrid HP-Mel map.
3. **Deep Feature Extraction:** Both branches utilize parallel VGG16 convolutional bases (with unfrozen weights dynamically fine-tuned at a low learning rate, $1e^{-5}$) to extract high-dimensional semantic vectors.
4. **Gated Attention Fusion:** To prevent feature redundancy, a trainable attention mechanism dynamically calculates gating weights using a Sigmoid activation, mathematically prioritizing continuous tonal features versus transient mode functions depending on the latent emotional class before feeding the fused vector to a Multi-Layer Perceptron (MLP) classifier.

## 📂 Repository Structure

```text
dde-speech-emotion-recognition/
│
├── data/
│   ├── raw_audio/                   # Place downloaded raw datasets here
│   │   ├── emodb/
│   │   └── ravdess/
│   └── precomputed_features/        # Extracted .npy tensors are saved here
│
├── modules/
│   ├── feature_extraction.py        # VMD and HP-Mel decomposition scripts
│   └── dde_architecture.py          # Dual-branch VGG16 and Gated Attention Layer
│
├── notebooks/
│   └── DDE_SER_Experiment.ipynb     # Main execution, augmentation, and LOSO loop
│
├── requirements.txt                 # Python dependencies
└── README.md
