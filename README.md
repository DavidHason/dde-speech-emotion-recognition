# DDE-SER: Dual-Decomposition Ensemble Framework for Speech Emotion Recognition

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Official implementation** of the paper: *"DDE-SER: A Dual-Decomposition Ensemble Framework Fusing Adaptive Variational Modes and Harmonic-Percussive Spectrograms for Speech Emotion Recognition"* by David Hason Rudd et al. (Submitted to *Human-centric Computing and Information Sciences*).

---

## 📖 Overview

Speech Emotion Recognition (SER) is persistently challenged by the non-stationary nature of vocal affect and transient, broadband noise. Standard deep learning pipelines often process spectrograms as monolithic inputs, failing to isolate orthogonal acoustic cues (e.g., separating tonal pitch from explosive consonants). 

**DDE-SER** is a novel, dual-branch deep learning framework that bridges 1D adaptive frequency decomposition with 2D structural time-frequency mapping. By utilizing **VGG-optiVMD** at the raw signal level and **Harmonic-Percussive (HP) decomposition** at the spectrogram level, the architecture successfully disentangles high-arousal emotional states, mitigating class confusion without requiring massive, computationally heavy transformer embeddings.

### ✨ Key Features & Methodology
* **Branch A (Adaptive Mode Extraction):** Utilizes VGG-optiVMD to autonomously extract optimal Intrinsic Mode Functions (IMFs). Fuses these into a multi-domain 3D spatial tensor comprising Mel-spectrograms, Chromagrams, and MFCCs.
* **Branch B (Structural Spectrogram Extraction):** Employs orthogonal median filtering to separate harmonic (tonal) and percussive (broadband noise) components, generating a highly structured 2D hybrid HP-Mel feature map.
* **Trainable Gated Attention Mechanism:** Dynamically weights the flattened feature vectors from both branches based on the latent emotional class to drastically mitigate feature redundancy.
* **Robust Acoustic Perturbation:** Implements targeted in-place data augmentation (pitch shifting and dynamic white Gaussian noise injection) to prevent overfitting on pristine studio corpora.
* **Explainable AI (XAI):** Provides Grad-CAM interpretability visualizations, proving mathematically and visually that the dual branches focus on distinct, complementary acoustic formants rather than dataset artifacts.

---

## 📊 Datasets and Performance

The framework was rigorously evaluated using a strict **Leave-One-Speaker-Out (LOSO)** cross-validation protocol to ensure true speaker-independence and zero data leakage.

| Dataset | Classes | Overall Accuracy (LOSO) | Macro F1-Score |
| :--- | :---: | :---: | :---: |
| **EMO-DB** (German) | 7 | **80.00%** | 0.73 |
| **RAVDESS** (English) | 8 | **61.00%** | 0.43 |

*Note: In our comprehensive comparative architectural benchmarking, the shallow, sequential convolutional structure of **VGG16** consistently outperformed deeper networks (ResNet50, EfficientNetB0) by avoiding severe overfitting on these relatively small acoustic datasets.*

---

## 📂 Repository Structure

```text
dde-speech-emotion-recognition/
│
├── data/
│   ├── raw_audio/                   # Raw .wav files (EMO-DB, RAVDESS)
│   └── precomputed_features/        # Extracted .npy tensors (Phase 3 Multi-Domain & HP-Mel)
│
├── modules/
│   ├── __init__.py
│   ├── feature_extraction.py        # VMD + Multi-domain fusion & HP filtering algorithms
│   └── dde_architecture.py          # Dual-branch VGG16 + Gated Attention model definition
│
├── notebooks/
│   ├── DDE_SER_Experiment_emodb.ipynb   # Execution loop and evaluation for EMO-DB
│   └── DDE_SER_Experiment_ravdess.ipynb # Execution loop and evaluation for RAVDESS
│
├── requirements.txt
└── README.md

