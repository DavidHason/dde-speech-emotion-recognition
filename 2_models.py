#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Author: David Hason Rudd
Description: Advanced Dual-Branch Architecture Definitions for SER.
Supports VGG16, ResNet50, and EfficientNetB0 backbones.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Multiply, Add, Dropout
from tensorflow.keras.applications import EfficientNetB0, ResNet50, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision

# Enable Mixed Precision
try:
    mixed_precision.set_global_policy('mixed_float16')
except AttributeError:
    policy = mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.experimental.set_policy(policy)

def squeeze_excite_block(input_tensor, ratio=16, name=""):
    """Squeeze-and-Excitation (SE) attention block[cite: 13]."""
    filters = input_tensor.shape[-1]
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False, name=f"{name}_se_squeeze")(input_tensor)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False, name=f"{name}_se_excite")(se)
    return Multiply(name=f"{name}_se_multiply")([input_tensor, se])

def build_advanced_dde_ser(input_shape=(128, 128, 3), num_classes=7, backbone="EfficientNetB0"):
    """Builds the Dual-Branch Deep Learning Engine[cite: 13]."""
    input_A = Input(shape=input_shape, name="vmd_input_A")
    input_B = Input(shape=input_shape, name="hp_input_B")
    
    # Select Backbone
    if backbone == "EfficientNetB0":
        base_A = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_A)
        base_B = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_B)
    elif backbone == "ResNet50":
        base_A = ResNet50(include_top=False, weights='imagenet', input_tensor=input_A)
        base_B = ResNet50(include_top=False, weights='imagenet', input_tensor=input_B)
    elif backbone == "VGG16":
        base_A = VGG16(include_top=False, weights='imagenet', input_tensor=input_A)
        base_B = VGG16(include_top=False, weights='imagenet', input_tensor=input_B)
        
    for layer in base_A.layers: layer._name = layer.name + "_A"
    for layer in base_B.layers: layer._name = layer.name + "_B"
        
    for layer in base_A.layers: layer.trainable = True
    for layer in base_B.layers: layer.trainable = True
        
    # Branch Processing
    feat_A = GlobalAveragePooling2D(name="gap_A")(base_A.output)
    feat_A = squeeze_excite_block(feat_A, name="branch_A")
    
    feat_B = GlobalAveragePooling2D(name="gap_B")(base_B.output)
    feat_B = squeeze_excite_block(feat_B, name="branch_B")
    
    # Fusion
    gate_A = Dense(feat_A.shape[-1], activation='sigmoid', name="gate_A")(feat_A)
    gate_B = Dense(feat_B.shape[-1], activation='sigmoid', name="gate_B")(feat_B)
    attended_A = Multiply(name="attended_A")([feat_A, gate_A])
    attended_B = Multiply(name="attended_B")([feat_B, gate_B])
    fused = Add(name="orthogonal_fusion")([attended_A, attended_B])
    
    # MLP Classifier
    x = Dropout(0.5)(fused)
    x = Dense(256, activation='selu', kernel_initializer='lecun_normal')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='selu', kernel_initializer='lecun_normal')(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32', name="emotion_output")(x)
    
    model = Model(inputs=[input_A, input_B], outputs=outputs, name=f"DDE_SER_{backbone}")
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

