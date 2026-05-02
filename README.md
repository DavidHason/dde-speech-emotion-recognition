# DDE-SER: A Dual-Decomposition Ensemble Framework with Dynamic Feature Fusion for Speech Emotion Recognition

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Official implementation** of the paper: *"DDE-SER: A Dual-Decomposition Ensemble Framework with Dynamic Feature Fusion for Speech Emotion Recognition"* by David Hason Rudd et al. (Submitted to *Human-centric Inteligent Systems (HCIS)*).

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

<img width="1762" height="3657" alt="DDE-SER" src="https://github.com/user-attachments/assets/5af59eb2-6ef7-46d0-98df-2425f8415472" />



## 📊 Datasets and Performance

The framework was rigorously evaluated using a strict **Leave-One-Speaker-Out (LOSO)** cross-validation protocol to ensure true speaker-independence and zero data leakage.

| Dataset | Classes | Overall Accuracy (LOSO) | Macro F1-Score |
| :--- | :---: | :---: | :---: |
| **EMO-DB** (German) | 7 | **82.55%** | 0.81 |
| **RAVDESS** (English) | 8 | **61.00%** | 0.59 |

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


⚙️ Installation and Requirements
The framework is built entirely in Python. It isolates heavy digital signal processing from the neural optimization loop to ensure computational efficiency and prevent GPU memory fragmentation.

1- Clone the repository:
git clone [https://github.com/DavidHason/dde-speech-emotion-recognition.git](https://github.com/DavidHason/dde-speech-emotion-recognition.git)
cd dde-speech-emotion-recognition

2- Install dependencies:
pip install -r requirements.txt


🚀 Usage
1. Dataset Access & Precomputed Features
Due to GitHub file size limits, the raw datasets and precomputed .npy arrays are hosted externally.

Download the Data: Access the EMO-DB and RAVDESS features here: Google Drive Dataset Link

Place the extracted contents directly into the data/precomputed_features/ directory.

2. Feature Extraction (Optional)
If you wish to extract features from scratch using your own .wav files, place them into the appropriate directories under data/raw_audio/ and run the extraction scripts in modules/feature_extraction.py.

Note: The VMD extraction performs Alternate Direction Method of Multipliers (ADMM) optimization and may take considerable time depending on dataset size.


3. Training and Evaluation (LOSO)
The training pipelines are located in the notebooks/ directory. The scripts handle:

Loading the precomputed .npy arrays.

Micro-batching (batch size 4) and dynamic VRAM allocation to manage the dense 3D inputs.

Executing the rigorous Leave-One-Speaker-Out cross-validation loop.

Automatically generating comprehensive classification reports, Area Under the Curve (AUC) metrics, and normalized Confusion Matrices (.tif format).

@article{HasonRudd2026DDESER,
  title={DDE-SER: A Dual-Decomposition Ensemble Framework Fusing Adaptive Variational Modes and Harmonic-Percussive Spectrograms for Speech Emotion Recognition},
  author={Hason Rudd, David and Wang, Xiangmeng and Islam, Md Rafiqul and Wang, Xianzhi and Sanin, Cesar and Huo, Huan and Xu, Guandong},
  journal={Human-centric Computing and Information Sciences},
  year={2026},


  note={Under Review}
}
