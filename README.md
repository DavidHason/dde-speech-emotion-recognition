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

📊 Datasets & Preprocessing
This framework was evaluated on two benchmark emotional speech corpora:

EMO-DB: Berlin Database of Emotional Speech (German, 7 classes)

RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song (English, 7 classes customized)

📥 Accessing the Data
To replicate this study, you must download the augmented and preprocessed datasets.

🔗 [Download the DDE-SER Datasets via Google Drive here] (INSERT_GOOGLE_DRIVE_LINK_HERE)

Once downloaded, extract the contents into the data/raw_audio/ directory.

🎛️ Acoustic Augmentation
To prevent overfitting caused by "blind oversampling" (exact duplication), our pipeline implements a scientifically robust acoustic perturbation methodology:

Pitch Shifting: Audio arrays are shifted upward by two half-steps to simulate natural inter-speaker vocal tract variations.

Noise Injection: Randomized, low-level additive white Gaussian noise (capped at 0.5% amplitude) is injected to simulate real-world acoustic environments.

(Note: The provided Google Drive link contains the base files. You can generate the augmented files dynamically using the augmentation cells provided in the Jupyter Notebook).

⚙️ Installation & Setup
Clone the repository:

git clone [https://github.com/DavidHason/dde-speech-emotion-recognition.git](https://github.com/DavidHason/dde-speech-emotion-recognition.git)
cd dde-speech-emotion-recognition

Set up a Python 3.10 virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

(Ensure you have CUDA/cuDNN configured for GPU acceleration via TensorFlow).

🚀 Usage
The entire pipeline is executable via the primary Jupyter Notebook located in notebooks/DDE_SER_Experiment.ipynb.

Run Feature Extraction: The script will traverse the raw_audio directories, parse the filenames, execute the VMD and HP decompositions, and save the resulting multi-dimensional tensors as .npy files to data/precomputed_features/.

Train the Model (LOSO): The training loop utilizes a strict Leave-One-Speaker-Out (LOSO) cross-validation protocol to ensure true generalizability. Class imbalances are handled dynamically via compute_class_weight.

Evaluate: The script automatically generates a classification report (Precision, Recall, Macro F1) and an academic, normalized Confusion Matrix (.tif format) specifically mapped to the dataset's emotion classes.

📝 Citation
If you utilize this code, methodology, or framework in your research, please cite our paper:

Code snippet
@article{HasonRudd2026DDESER,
  title={DDE-SER: A Dual-Decomposition Ensemble Framework Fusing Adaptive Variational Modes and Harmonic-Percussive Spectrograms for Speech Emotion Recognition},
  author={Hason Rudd, David and Islam, MD Rafiqul and Wang, Xianzhi and Sanin, Cesar and Huo, Huan and Xu, Guandong},
  journal={Tech Science Press},
  year={2026}
}

### Additional Suggestions for Your Repository:

1. **Create a `requirements.txt` file:** Include libraries like `tensorflow==2.10.0`, `librosa`, `vmdpy`, `scikit-learn`, `seaborn`, `matplotlib`, and `tqdm`. This ensures anyone trying to replicate your work uses the exact same versions.
2. **Add a `LICENSE` file:** Since this is an academic project, adding an MIT or Apache 2.0 license file to your GitHub repository is highly recommended so other researchers know they can legally build upon your code.
3. **Zenodo Integration:** While a Google Drive link is great for immediate sharing,
