#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Author: David Hason Rudd
Description: Training execution script for DDE-SER.
Implements Leave-One-Group-Out (LOSO) cross validation and timing callbacks.
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
from models import build_advanced_dde_ser

class EpochTimingCallback(tf.keras.callbacks.Callback):
    """Callback to measure the exact time it takes to train each epoch[cite: 13]."""
    def on_train_begin(self, logs=None):
        self.epoch_times = []
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.time() - self.epoch_start_time)

def run_comparative_study(dataset_name, X_A, X_B, Y, speakers):
    """Iterates through backbones, runs LOSO, and exports results[cite: 13]."""
    print(f"========== STARTING COMPARATIVE STUDY: {dataset_name} ==========")
    backbones_to_test = ["VGG16", "ResNet50", "EfficientNetB0"]
    num_classes = Y.shape[1]
    comparative_results = []
    logo = LeaveOneGroupOut()

    try:
        AUTOTUNE = tf.data.AUTOTUNE
    except AttributeError:
        AUTOTUNE = tf.data.experimental.AUTOTUNE

    for backbone in backbones_to_test:
        print(f"\\n>>> INITIALIZING BACKBONE: {backbone} <<<")
        
        all_true = []
        all_preds = []
        fold_accuracies = []
        backbone_epoch_times = []
        
        for fold, (train_idx, test_idx) in enumerate(logo.split(X_A, Y, groups=speakers)):
            print(f"  -> Training Fold {fold + 1} / {len(np.unique(speakers))}...", end="")
            
            X_A_train, X_A_test = X_A[train_idx], X_A[test_idx]
            X_B_train, X_B_test = X_B[train_idx], X_B[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            y_ints = np.argmax(Y_train, axis=1)
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_ints), y=y_ints)
            class_weight_dict = dict(enumerate(class_weights))
            
            train_dataset = tf.data.Dataset.from_tensor_slices(((X_A_train, X_B_train), Y_train))
            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(8).prefetch(AUTOTUNE)
            
            val_dataset = tf.data.Dataset.from_tensor_slices(((X_A_test, X_B_test), Y_test))
            val_dataset = val_dataset.batch(8).prefetch(AUTOTUNE)
            
            model = build_advanced_dde_ser(num_classes=num_classes, backbone=backbone)
            
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            time_callback = EpochTimingCallback()
            
            model.fit(train_dataset, validation_data=val_dataset,
                      epochs=100, class_weight=class_weight_dict,
                      callbacks=[early_stop, time_callback], verbose=0)
            
            loss, acc = model.evaluate(val_dataset, verbose=0)
            fold_accuracies.append(acc)
            backbone_epoch_times.extend(time_callback.epoch_times)
            
            probs = model.predict(val_dataset, verbose=0)
            all_preds.extend(np.argmax(probs, axis=1))
            all_true.extend(np.argmax(Y_test, axis=1))
            
            print(f" Done (Acc: {acc:.4f})")
            
            del model, X_A_train, X_A_test, X_B_train, X_B_test, Y_train, Y_test
            del train_dataset, val_dataset
            tf.keras.backend.clear_session()
            gc.collect()

        overall_acc = accuracy_score(all_true, all_preds)
        macro_f1 = f1_score(all_true, all_preds, average='macro')
        avg_epoch_time = np.mean(backbone_epoch_times)
        
        print(f"\\n[RESULT] {backbone} | Acc: {overall_acc:.4f} | F1: {macro_f1:.4f} | Avg Time/Epoch: {avg_epoch_time:.2f}s")
        
        comparative_results.append({
            "Dataset": dataset_name,
            "Architecture": backbone,
            "Overall_Accuracy": overall_acc,
            "Macro_F1_Score": macro_f1,
            "Avg_Epoch_Time_Sec": avg_epoch_time
        })
        
        df_results = pd.DataFrame(comparative_results)
        df_results.to_csv(f"Comparative_Study_{dataset_name}.csv", index=False)
        
    print(f"\\n========== {dataset_name} STUDY COMPLETE ==========")
    return df_results

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    gc.collect()

    print("Loading Precomputed RAVDESS Features...")
    X_A_rav = np.load('data/precomputed_features/ravdess_vmd_128.npy') 
    X_B_rav = np.load('data/precomputed_features/ravdess_hp_128.npy')
    Y_rav = np.load('data/precomputed_features/ravdess_labels.npy')
    speakers_rav = np.load('data/precomputed_features/ravdess_speakers.npy')

    # Proper ImageNet Scaling [0, 255][cite: 13]
    X_A_rav = (X_A_rav - np.min(X_A_rav)) / (np.max(X_A_rav) - np.min(X_A_rav) + 1e-8) * 255.0
    X_B_rav = (X_B_rav - np.min(X_B_rav)) / (np.max(X_B_rav) - np.min(X_B_rav) + 1e-8) * 255.0

    # Execute study[cite: 13]
    ravdess_results = run_comparative_study("RAVDESS", X_A_rav, X_B_rav, Y_rav, speakers_rav)
    print(ravdess_results)

