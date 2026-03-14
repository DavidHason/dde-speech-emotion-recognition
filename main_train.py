"""
DDE-SER: Dual-Decomposition Ensemble Framework
Main Training and Evaluation Script
"""

import os
import gc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_class_weight

# Import custom architecture
from modules.dde_architecture import build_dde_ser_model

def setup_gpu():
    """Forces TensorFlow to dynamically allocate GPU memory to prevent OOM crashes."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[INFO] GPU Memory Growth Enabled.")
        except RuntimeError as e:
            print(f"[ERROR] GPU Setup Failed: {e}")

def load_and_normalize_data(dataset_name="RAVDESS", base_path="data/precomputed_features/"):
    """Loads precomputed .npy tensors and applies Min-Max Normalization."""
    print(f"\n[INFO] Loading {dataset_name} features from disk...")
    
    prefix = dataset_name.lower().replace("-", "")
    
    X_A = np.load(os.path.join(base_path, f'{prefix}_vmd_128.npy'))
    X_B = np.load(os.path.join(base_path, f'{prefix}_hp_128.npy'))
    Y = np.load(os.path.join(base_path, f'{prefix}_labels.npy'))
    speaker_ids = np.load(os.path.join(base_path, f'{prefix}_speakers.npy'))

    print("[INFO] Applying Min-Max Normalization [0, 1]...")
    X_A = (X_A - np.min(X_A)) / (np.max(X_A) - np.min(X_A) + 1e-8)
    X_B = (X_B - np.min(X_B)) / (np.max(X_B) - np.min(X_B) + 1e-8)

    print(f"       Branch A Shape: {X_A.shape}")
    print(f"       Branch B Shape: {X_B.shape}")
    print(f"       Labels Shape: {Y.shape}")
    
    return X_A, X_B, Y, speaker_ids

def execute_loso_training(X_A, X_B, Y, speaker_ids):
    """Executes the Leave-One-Speaker-Out (LOSO) cross-validation loop."""
    logo = LeaveOneGroupOut()
    fold_accuracies = []
    all_true_labels = []
    all_predictions = []

    print("\n[INFO] Commencing Leave-One-Speaker-Out (LOSO) Training...")
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X_A, Y, groups=speaker_ids)):
        print(f"--- Starting LOSO Fold {fold + 1} ---")
        
        X_A_train, X_A_test = X_A[train_idx], X_A[test_idx]
        X_B_train, X_B_test = X_B[train_idx], X_B[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        # DYNAMIC CLASS WEIGHTING
        y_integers = np.argmax(Y_train, axis=1)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
        class_weight_dict = dict(enumerate(class_weights))
        
        # Build Model
        model = build_dde_ser_model(num_classes=Y.shape[1]) 
        
        # Callbacks
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'best_model_fold_{fold+1}.h5', 
            monitor='val_accuracy', 
            save_best_only=True, 
            save_weights_only=True, 
            verbose=0
        )
        
        # Train
        model.fit(
            [X_A_train, X_B_train], Y_train,
            validation_data=([X_A_test, X_B_test], Y_test),
            epochs=100, 
            batch_size=32,
            class_weight=class_weight_dict, 
            callbacks=[early_stop, checkpoint],
            verbose=0 
        )
        
        # Evaluate
        loss, accuracy = model.evaluate([X_A_test, X_B_test], Y_test, verbose=0)
        fold_accuracies.append(accuracy)
        print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")
        
        # Store predictions
        preds = np.argmax(model.predict([X_A_test, X_B_test], verbose=0), axis=1)
        trues = np.argmax(Y_test, axis=1)
        all_true_labels.extend(trues)
        all_predictions.extend(preds)
        
        # Memory Cleanup
        del model, X_A_train, X_A_test, X_B_train, X_B_test, Y_train, Y_test
        tf.keras.backend.clear_session()
        gc.collect()

    print(f"\n[RESULT] Overall LOSO Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    return all_true_labels, all_predictions

def evaluate_and_visualize(true_labels, predictions, dataset_name):
    """Generates the classification report and saves the confusion matrix."""
    print(f"\n========== {dataset_name} EVALUATION METRICS ==========")
    
    if dataset_name == "EMO-DB":
        emotion_labels = ['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Neutral']
    else: # RAVDESS
        emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness']

    # Text Report
    report = classification_report(true_labels, predictions, target_names=emotion_labels)
    print(report)

    # Confusion Matrix Visualization
    cm = confusion_matrix(true_labels, predictions)
    row_sums = cm.sum(axis=1)
    row_sums[row_sums == 0] = 1 
    cm_normalized = cm.astype('float') / row_sums[:, np.newaxis]

    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=emotion_labels, yticklabels=emotion_labels,
                cbar_kws={'label': 'Prediction Probability'})

    plt.title(f'DDE-SER Model: {dataset_name} LOSO Confusion Matrix', fontsize=16, pad=15)
    plt.ylabel('True Emotion', fontsize=14)
    plt.xlabel('Predicted Emotion', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'DDE_SER_Confusion_Matrix_{dataset_name}.tif', dpi=300, format='tif')
    print(f"[INFO] Confusion matrix saved as DDE_SER_Confusion_Matrix_{dataset_name}.tif")

if __name__ == "__main__":
    # 1. Set your target dataset here ("RAVDESS" or "EMO-DB")
    TARGET_DATASET = "RAVDESS" 
    
    setup_gpu()
    
    try:
        X_A, X_B, Y, speakers = load_and_normalize_data(dataset_name=TARGET_DATASET)
        trues, preds = execute_loso_training(X_A, X_B, Y, speakers)
        evaluate_and_visualize(trues, preds, TARGET_DATASET)
    except FileNotFoundError:
        print("[ERROR] Data files not found. Please run the feature_extraction.py module first.")