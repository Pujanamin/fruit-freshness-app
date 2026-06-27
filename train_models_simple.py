"""
Quick Training Script - Trains both models from Kaggle dataset.
Run this ONCE to generate trained models for production use.

Usage:
    python train_models_simple.py
"""

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Sequential, applications
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import config

print("=" * 70)
print("FRUIT FRESHNESS MODEL TRAINING")
print("=" * 70)

# ============================================================================
# STEP 1: DOWNLOAD & PREPARE DATASET
# ============================================================================
print("\n[STEP 1] Downloading Kaggle Fruits Dataset...")
try:
    import kagglehub
    dataset_path = kagglehub.dataset_download('shivamardeshna/fruits-dataset')
    print(f"✓ Dataset downloaded to: {dataset_path}")
except Exception as e:
    print(f"⚠ Download failed: {e}")
    print("  Please manually download from: https://www.kaggle.com/datasets/shivamardeshna/fruits-dataset")
    dataset_path = input("Enter dataset path: ")

# ============================================================================
# STEP 2: LOAD DATASET FROM SUBDIRECTORIES
# ============================================================================
print("\n[STEP 2] Loading dataset...")

# Find the train/val/test directories
for root, dirs, files in os.walk(dataset_path):
    if 'train' in root.lower() and dirs:
        train_dir = root
        break

# Assuming standard Kaggle structure with train/validation/test splits
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    seed=42
)
print(f"✓ Training dataset loaded: {train_dataset}")

# If validation/test exist
try:
    val_dir = train_dir.replace('train', 'validation')
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        seed=42
    )
    print(f"✓ Validation dataset loaded")
except:
    print("⚠ Validation dataset not found, using 20% of training as validation")
    total_samples = len(train_dataset) * BATCH_SIZE
    val_size = int(0.2 * total_samples)
    val_dataset = train_dataset.take(val_size // BATCH_SIZE)
    train_dataset = train_dataset.skip(val_size // BATCH_SIZE)

# ============================================================================
# STEP 3: BUILD & TRAIN CUSTOM CNN
# ============================================================================
print("\n[STEP 3A] Building Custom CNN Model...")

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=6):
    """Simple custom CNN for fruit classification."""
    model = Sequential([
        layers.Input(shape=input_shape),
        
        # Normalization
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        
        # Block 1
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),
        
        # Block 2
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.4),
        
        # Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='Custom_CNN')
    
    return model

custom_model = build_custom_cnn()
custom_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model built. Training...")
print("\n[STEP 3B] Training Custom CNN (max 20 epochs)...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

history_custom = custom_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

# Save Custom CNN
os.makedirs('models', exist_ok=True)
custom_model.save('models/custom_cnn_fruit_classifier.h5')
print(f"✓ Custom CNN saved to: models/custom_cnn_fruit_classifier.h5")

# ============================================================================
# STEP 4: BUILD & TRAIN TRANSFER LEARNING (EfficientNetB0)
# ============================================================================
print("\n[STEP 4A] Building Transfer Learning Model (EfficientNetB0)...")

def build_transfer_learning_model(input_shape=(224, 224, 3), num_classes=6):
    """Transfer Learning with EfficientNetB0 backbone."""
    base = applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False  # Freeze backbone
    
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ], name='Transfer_Learning_EfficientNetB0')
    
    return model

transfer_model = build_transfer_learning_model()
transfer_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model built. Training...")
print("\n[STEP 4B] Training Transfer Learning (max 15 epochs)...")

history_transfer = transfer_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# Save Transfer Learning Model
transfer_model.save('models/transfer_learning_fruit_classifier.h5')
print(f"✓ Transfer Learning saved to: models/transfer_learning_fruit_classifier.h5")

# ============================================================================
# STEP 5: EVALUATION & SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)

print(f"\n✓ Custom CNN final val accuracy: {history_custom.history['val_accuracy'][-1]*100:.1f}%")
print(f"✓ Transfer Learning final val accuracy: {history_transfer.history['val_accuracy'][-1]*100:.1f}%")

print("\nModels saved:")
print(f"  - {os.path.abspath('models/custom_cnn_fruit_classifier.h5')}")
print(f"  - {os.path.abspath('models/transfer_learning_fruit_classifier.h5')}")

print("\nNext: Restart Streamlit to use trained models!")
print("  Command: streamlit run app.py")
print("=" * 70)
