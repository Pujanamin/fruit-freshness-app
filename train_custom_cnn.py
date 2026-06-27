"""
================================================================================
                    TRAIN_CUSTOM_CNN.PY
        Training Script for Custom CNN Fruit Classifier
================================================================================
Purpose:
    Train a custom-built convolutional neural network from scratch on the 
    Kaggle Fruits Dataset. Includes architecture definition, training loop,
    model evaluation, and checkpoint saving.

Usage:
    python train_custom_cnn.py

Output:
    - Trained model: models/custom_cnn_fruit_classifier.h5
    - Training history: models/training_history_custom_cnn.pkl
    - Metrics report: models/metrics_custom_cnn.json
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
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import classification_report, confusion_matrix
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")


# ============================================================================
# CUSTOM CNN ARCHITECTURE
# ============================================================================
def build_custom_cnn_model(
    input_shape: tuple = (config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH, 3),
    num_classes: int = config.NUM_CLASSES,
) -> models.Model:
    """
    Build custom CNN architecture optimized for fruit classification.
    
    Architecture:
    - Input: 224×224×3
    - Block 1: Conv(32) → Conv(32) → MaxPool
    - Block 2: Conv(64) → Conv(64) → MaxPool
    - Block 3: Conv(128) → Conv(128) → MaxPool
    - Global Average Pooling
    - Dense(256, ReLU, Dropout)
    - Dense(num_classes, Softmax)
    
    Args:
        input_shape: Input tensor shape
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Block 1: 32 filters
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2: 64 filters
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3: 128 filters
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4: 256 filters
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Global pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


# ============================================================================
# TRAINING PIPELINE
# ============================================================================
def train_custom_cnn():
    """Main training pipeline for Custom CNN model."""
    
    print("\n" + "="*70)
    print("CUSTOM CNN FRUIT CLASSIFIER - TRAINING PIPELINE")
    print("="*70)
    
    if not TENSORFLOW_AVAILABLE:
        print("ERROR: TensorFlow not available")
        return
    
    # Setup
    print("\n[1/6] Setting up dataset directory...")
    dataset_dir = setup_dataset_directory()
    
    # Load dataset
    print("\n[2/6] Loading dataset...")
    loader = FruitsDatasetLoader(dataset_dir)
    images, labels = loader.load_dataset()
    
    # Create splits
    print("\n[3/6] Creating train/val/test splits...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        loader.create_train_val_test_split(images, labels)
    
    # Normalize images
    print("\n[4/6] Normalizing images...")
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Convert labels to one-hot
    y_train_onehot = keras.utils.to_categorical(y_train, config.NUM_CLASSES)
    y_val_onehot = keras.utils.to_categorical(y_val, config.NUM_CLASSES)
    y_test_onehot = keras.utils.to_categorical(y_test, config.NUM_CLASSES)
    
    # Build model
    print("\n[5/6] Building model...")
    model = build_custom_cnn_model()
    
    print("\nModel Architecture:")
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(config.MODELS_DIR / "custom_cnn_fruit_classifier.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Training
    print("\n[6/6] Training model...")
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
    
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=50,
        batch_size=config.MAX_BATCH_SIZE,
        callbacks=[checkpoint_callback, early_stopping, reduce_lr],
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
        "model_type": "Custom CNN",
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
    }
    
    metrics_path = config.MODELS_DIR / "metrics_custom_cnn.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Save training history
    history_path = config.MODELS_DIR / "training_history_custom_cnn.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"✓ Training history saved to: {history_path}")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"Model saved: {config.MODELS_DIR / 'custom_cnn_fruit_classifier.h5'}")
    print(f"Model size: {os.path.getsize(config.MODELS_DIR / 'custom_cnn_fruit_classifier.h5') / (1024*1024):.2f} MB")


if __name__ == "__main__":
    train_custom_cnn()
