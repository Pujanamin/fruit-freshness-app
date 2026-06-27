"""
================================================================================
                    TRAIN_TRANSFER_LEARNING.PY
        Training Script for Transfer Learning Fruit Classifier
================================================================================
Purpose:
    Train an EfficientNetB0-based transfer learning model on the 
    Kaggle Fruits Dataset. Uses pre-trained ImageNet weights and fine-tunes
    for fruit classification. Generally achieves higher accuracy than custom CNN.

Usage:
    python train_transfer_learning.py

Output:
    - Trained model: models/transfer_learning_fruit_classifier.h5
    - Training history: models/training_history_transfer_learning.pkl
    - Metrics report: models/metrics_transfer_learning.json
================================================================================
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

import config
from dataset_loader import FruitsDatasetLoader, setup_dataset_directory

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import classification_report, confusion_matrix
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")


# ============================================================================
# TRANSFER LEARNING MODEL ARCHITECTURE
# ============================================================================
def build_transfer_learning_model(
    input_shape: tuple = (config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH, 3),
    num_classes: int = config.NUM_CLASSES,
    trainable_layers: int = -50,  # Unfreeze last 50 layers for fine-tuning
) -> models.Model:
    """
    Build EfficientNetB0-based transfer learning model.
    
    Architecture:
    - Base: EfficientNetB0 (pre-trained on ImageNet)
    - Frozen backbone layers (except last trainable_layers)
    - Global Average Pooling
    - Dense(256, ReLU, Dropout) - Custom head
    - Dense(num_classes, Softmax)
    
    Args:
        input_shape: Input tensor shape
        num_classes: Number of output classes
        trainable_layers: Number of layers from end to unfreeze (-50 means last 50)
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers except the last ones
    total_layers = len(base_model.layers)
    freeze_until = total_layers + trainable_layers  # trainable_layers is negative
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    print(f"\n✓ Frozen {freeze_until} base layers")
    print(f"✓ Fine-tuning last {total_layers - freeze_until} layers")
    
    # Build complete model
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Pre-trained base
        base_model,
        
        # Custom classification head
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


# ============================================================================
# TWO-STAGE TRAINING: FEATURE EXTRACTION + FINE-TUNING
# ============================================================================
def train_transfer_learning():
    """Main training pipeline with two-stage approach."""
    
    print("\n" + "="*70)
    print("TRANSFER LEARNING FRUIT CLASSIFIER - TRAINING PIPELINE")
    print("EfficientNetB0 Base with Fine-tuning")
    print("="*70)
    
    if not TENSORFLOW_AVAILABLE:
        print("ERROR: TensorFlow not available")
        return
    
    # Setup
    print("\n[1/7] Setting up dataset directory...")
    dataset_dir = setup_dataset_directory()
    
    # Load dataset
    print("\n[2/7] Loading dataset...")
    loader = FruitsDatasetLoader(dataset_dir)
    images, labels = loader.load_dataset()
    
    # Create splits
    print("\n[3/7] Creating train/val/test splits...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        loader.create_train_val_test_split(images, labels)
    
    # Normalize images (ImageNet normalization)
    print("\n[4/7] Normalizing images...")
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Convert labels to one-hot
    y_train_onehot = keras.utils.to_categorical(y_train, config.NUM_CLASSES)
    y_val_onehot = keras.utils.to_categorical(y_val, config.NUM_CLASSES)
    y_test_onehot = keras.utils.to_categorical(y_test, config.NUM_CLASSES)
    
    # Build model
    print("\n[5/7] Building transfer learning model...")
    model = build_transfer_learning_model()
    
    print("\nModel Architecture (first 20 layers):")
    for i, layer in enumerate(model.layers[:20]):
        print(f"  {i}: {layer.name:30s} {layer.output_shape}")
    
    # Stage 1: Feature extraction (frozen base)
    print("\n[6/7a] STAGE 1 - Feature Extraction (Frozen Backbone)...")
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    checkpoint_stage1 = keras.callbacks.ModelCheckpoint(
        filepath=str(config.MODELS_DIR / ".tmp_model_stage1.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping_stage1 = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    history_stage1 = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=20,
        batch_size=config.MAX_BATCH_SIZE,
        callbacks=[checkpoint_stage1, early_stopping_stage1],
        verbose=1
    )
    
    # Stage 2: Fine-tuning (unfreeze base layers)
    print("\n[6/7b] STAGE 2 - Fine-tuning (Unfrozen Backbone)...")
    
    # Unfreeze base model for fine-tuning
    for layer in model.layers:
        layer.trainable = True
    
    # Compile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    checkpoint_stage2 = keras.callbacks.ModelCheckpoint(
        filepath=str(config.MODELS_DIR / "transfer_learning_fruit_classifier.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping_stage2 = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    history_stage2 = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=30,
        batch_size=config.MAX_BATCH_SIZE,
        callbacks=[checkpoint_stage2, early_stopping_stage2, reduce_lr],
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Predictions for detailed metrics
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=config.FRUIT_CLASS_NAMES
    ))
    
    # Save metrics
    metrics = {
        "model_type": "Transfer Learning (EfficientNetB0)",
        "training_date": datetime.now().isoformat(),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=config.FRUIT_CLASS_NAMES,
            output_dict=True
        ),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "base_model": "EfficientNetB0",
        "training_stages": 2,
    }
    
    metrics_path = config.MODELS_DIR / "metrics_transfer_learning.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Save combined training history
    combined_history = {
        "stage1": history_stage1.history,
        "stage2": history_stage2.history,
    }
    history_path = config.MODELS_DIR / "training_history_transfer_learning.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(combined_history, f)
    print(f"✓ Training history saved to: {history_path}")
    
    # Cleanup
    tmp_file = config.MODELS_DIR / ".tmp_model_stage1.h5"
    if tmp_file.exists():
        os.remove(tmp_file)
    
    print("\n" + "="*70)
    print("[7/7] ✓ TRAINING COMPLETE")
    print("="*70)
    print(f"Model saved: {config.MODELS_DIR / 'transfer_learning_fruit_classifier.h5'}")
    print(f"Model size: {os.path.getsize(config.MODELS_DIR / 'transfer_learning_fruit_classifier.h5') / (1024*1024):.2f} MB")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")


if __name__ == "__main__":
    train_transfer_learning()
