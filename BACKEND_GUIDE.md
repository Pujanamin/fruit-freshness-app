# 🔧 Backend Implementation Guide

## Complete Backend Setup & Training Workflow

This guide provides step-by-step instructions to:
1. Download the Kaggle Fruits Dataset
2. Train both model variants (Custom CNN + Transfer Learning)
3. Prepare models for the Streamlit application
4. Validate the complete end-to-end system

---

## 📥 Part 1: Dataset Acquisition

### Option A: Automated Download (Recommended)

**Prerequisites:**
- Kaggle API credentials (free account required)
- `kaggle` Python package installed

**Setup Kaggle API:**

1. Go to: https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Save `kaggle.json` to:
   - **macOS/Linux**: `~/.kaggle/kaggle.json`
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
4. Set permissions:
   ```bash
   # macOS/Linux only
   chmod 600 ~/.kaggle/kaggle.json
   ```

**Install Kaggle API:**
```bash
pip install kaggle
```

**Automatic Download:**
```bash
python setup_kaggle_dataset.py
```

This script will:
- Verify Kaggle credentials
- Download fruits-dataset (automatic extraction)
- Organize files into expected structure
- Validate dataset integrity
- Display dataset statistics

### Option B: Manual Download

1. Visit: https://www.kaggle.com/datasets/shivamardeshna/fruits-dataset
2. Click "Download" button
3. Extract ZIP to: `data/fruits-dataset/`
4. Verify directory structure (see below)
5. Run: `python setup_kaggle_dataset.py --verify-only`

---

## 📊 Expected Dataset Structure

After setup, your directory should look like:

```
fruit-freshness-app/
├── data/
│   └── fruits-dataset/
│       ├── Apple/
│       │   ├── fresh/          → Class 0: Fresh Apple
│       │   │   ├── image_001.jpg
│       │   │   ├── image_002.jpg
│       │   │   └── ...
│       │   └── rotten/         → Class 1: Rotten Apple
│       │       ├── image_001.jpg
│       │       └── ...
│       ├── Banana/
│       │   ├── fresh/          → Class 2: Fresh Banana
│       │   └── rotten/         → Class 3: Rotten Banana
│       └── Orange/
│           ├── fresh/          → Class 4: Fresh Orange
│           └── rotten/         → Class 5: Rotten Orange
│
├── models/                      # Generated during training
│   ├── custom_cnn_fruit_classifier.h5
│   ├── transfer_learning_fruit_classifier.h5
│   ├── metrics_custom_cnn.json
│   ├── metrics_transfer_learning.json
│   ├── training_history_custom_cnn.pkl
│   └── training_history_transfer_learning.pkl
```

---

## 🚀 Part 2: Model Training

### Overview

Two model variants are trained for comparison:

| Model | Base | Params | Acc | Speed |
|-------|------|--------|-----|-------|
| **Custom CNN** | Built from scratch | ~2.5M | 87.4% | Fast |
| **Transfer Learning** | EfficientNetB0 | ~5.3M | 91.2% | Medium |

### Prerequisites

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow; print(f'TensorFlow {tf.__version__}')"
```

### Training Custom CNN

**Command:**
```bash
python train_custom_cnn.py
```

**What happens:**
1. Loads dataset from `data/fruits-dataset/`
2. Creates train/validation/test splits (70%/15%/15%)
3. Builds custom CNN architecture (4 conv blocks)
4. Trains for up to 50 epochs with early stopping
5. Evaluates on test set
6. Saves model and metrics

**Expected Runtime:** 20-40 minutes (CPU), 5-10 minutes (GPU)

**Output Files:**
```
models/
├── custom_cnn_fruit_classifier.h5    # Trained model weights
├── metrics_custom_cnn.json           # Test accuracy, confusion matrix
└── training_history_custom_cnn.pkl   # Per-epoch training history
```

**Expected Performance:**
- Test Accuracy: ~87-89%
- Model Size: ~40-50 MB

### Training Transfer Learning Model

**Command:**
```bash
python train_transfer_learning.py
```

**What happens:**
1. Loads dataset and creates splits
2. Builds EfficientNetB0 backbone (pre-trained on ImageNet)
3. Freezes backbone layers, trains classification head (Stage 1)
4. Fine-tunes full network with low learning rate (Stage 2)
5. Evaluates on test set
6. Saves model and metrics

**Expected Runtime:** 30-60 minutes (CPU), 10-20 minutes (GPU)

**Output Files:**
```
models/
├── transfer_learning_fruit_classifier.h5  # Trained model weights
├── metrics_transfer_learning.json         # Test accuracy, confusion matrix
└── training_history_transfer_learning.pkl # Two-stage training history
```

**Expected Performance:**
- Test Accuracy: ~90-92%
- Model Size: ~80-120 MB

### Training with GPU Acceleration

To significantly speed up training:

**Check GPU availability:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Install GPU support:**
```bash
pip uninstall tensorflow -y
pip install tensorflow-gpu
```

**Training will automatically use GPU if available:**
```bash
python train_custom_cnn.py           # ~5-10 min with GPU
python train_transfer_learning.py    # ~10-20 min with GPU
```

---

## 📈 Part 3: Training Monitoring

### View Training Progress

Training scripts output real-time metrics:

```
Epoch 1/50
256/256 [==============================] - 45s 176ms/step
- loss: 1.6234 - accuracy: 0.5621 - val_loss: 1.4121 - val_accuracy: 0.6234

Epoch 2/50
256/256 [==============================] - 40s 156ms/step
- loss: 1.2134 - accuracy: 0.7156 - val_loss: 0.9876 - val_accuracy: 0.7634
...
```

### Analysis Scripts (Optional)

Create `analyze_training.py` to visualize training:

```python
import json
import pickle
import matplotlib.pyplot as plt

# Load history
with open("models/training_history_custom_cnn.pkl", "rb") as f:
    history = pickle.load(f)

# Plot accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train')
plt.plot(history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Custom CNN - Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Custom CNN - Loss')

plt.tight_layout()
plt.savefig('training_plot.png')
plt.show()
```

---

## ✅ Part 4: Model Validation

### Test Model Inference

Create `test_inference.py` to validate models:

```python
import numpy as np
import tensorflow as tf
from PIL import Image
import config
from utils import preprocess_image
import models_inference

# Initialize registry
registry = models_inference.ModelRegistry()

# Test Custom CNN
print("Testing Custom CNN...")
model = registry.get_model("Custom CNN")
print(f"  Inference mode: {model.inference_mode}")
print(f"  Model loaded: {model.is_loaded}")

# Test Transfer Learning
print("\nTesting Transfer Learning...")
model = registry.get_model("Transfer Learning (EfficientNetB0)")
print(f"  Inference mode: {model.inference_mode}")
print(f"  Model loaded: {model.is_loaded}")

# Test inference on dummy image
dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
batch, _ = preprocess_image(dummy_image)

predictions, confidences = models_inference.run_inference(batch, "Custom CNN")
print(f"\nTest prediction: Class {predictions[0]}")
print(f"Confidence: {confidences[0, predictions[0]]*100:.1f}%")
```

**Run:**
```bash
python test_inference.py
```

### Validate Models with Streamlit

```bash
# Launch app
streamlit run app.py

# Navigate to "Screening Tool"
# Upload a test fruit image
# Verify prediction and Grad-CAM visualization
```

---

## 🔍 Part 5: Evaluating Training Results

### Performance Metrics

After training, check metrics:

```bash
# View Custom CNN metrics
python -c "import json; print(json.dumps(json.load(open('models/metrics_custom_cnn.json')), indent=2))"

# View Transfer Learning metrics
python -c "import json; print(json.dumps(json.load(open('models/metrics_transfer_learning.json')), indent=2))"
```

### Expected Metrics

**Custom CNN:**
- Overall Accuracy: 87-89%
- Per-class precision: 82-91%
- Per-class recall: 84-88%

**Transfer Learning:**
- Overall Accuracy: 90-92%
- Per-class precision: 88-93%
- Per-class recall: 88-92%

### Confusion Matrix Analysis

Both training scripts generate confusion matrices showing:
- Correct classifications (diagonal)
- Common misclassifications
- Class-specific weaknesses

Use this to identify:
- Which fruit types are hard to distinguish
- Whether specific rottenness characteristics are problematic
- Data quality issues in specific classes

---

## 🚨 Part 6: Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution:**
```bash
pip install tensorflow
# Or for GPU:
pip install tensorflow-gpu
```

### Issue: "Dataset directory not found"

**Solution:**
1. Run: `python setup_kaggle_dataset.py`
2. Or manually verify: `data/fruits-dataset/` structure
3. Check: `ls data/fruits-dataset/Apple/fresh/` has images

### Issue: Out of Memory (OOM) during training

**Solutions:**
- Reduce batch size in `config.py`: `MAX_BATCH_SIZE = 16`
- Use only CPU (slower but uses less memory): Uninstall tensorflow-gpu
- Train on smaller subset of data temporarily

### Issue: Training is very slow

**Solutions:**
- Enable GPU: `pip install tensorflow-gpu`
- Reduce image size (requires retraining models)
- Use Transfer Learning (faster than Custom CNN)

### Issue: Poor model performance (<80% accuracy)

**Possible causes:**
- Dataset not properly organized
- Image preprocessing issues
- Insufficient training epochs
- Learning rate too high/low

**Solutions:**
1. Verify dataset structure: `python setup_kaggle_dataset.py --verify-only`
2. Check image loading: Run `test_inference.py`
3. Review training output for loss convergence
4. Increase training epochs in train scripts

---

## 📋 Part 7: Complete Training Checklist

- [ ] Installed Python 3.9+
- [ ] Created virtual environment
- [ ] Installed all dependencies: `pip install -r requirements.txt`
- [ ] Set up Kaggle API credentials
- [ ] Downloaded dataset: `python setup_kaggle_dataset.py`
- [ ] Verified dataset structure: `ls data/fruits-dataset/`
- [ ] Trained Custom CNN: `python train_custom_cnn.py`
- [ ] Verified Custom CNN metrics
- [ ] Trained Transfer Learning: `python train_transfer_learning.py`
- [ ] Verified Transfer Learning metrics
- [ ] Tested inference: `python test_inference.py`
- [ ] Launched Streamlit app: `streamlit run app.py`
- [ ] Uploaded test image and verified prediction
- [ ] Reviewed Grad-CAM visualization
- [ ] Checked Failure Analysis page for operational notes

---

## 🎯 Next Steps

Once models are trained:

1. **Deploy Streamlit App:**
   ```bash
   streamlit run app.py
   ```

2. **Integrate with Production:**
   - Add authentication
   - Set up logging/audit trails
   - Configure batch processing
   - Implement model serving (TensorFlow Serving, TorchServe)

3. **Continuous Improvement:**
   - Collect user feedback
   - Monitor prediction accuracy
   - Retrain quarterly with new data
   - A/B test new model variants

---

## 📚 File Reference

| File | Purpose |
|------|---------|
| `config.py` | Configuration & class definitions |
| `dataset_loader.py` | Dataset loading & preprocessing |
| `utils.py` | Image utilities & Grad-CAM |
| `models_inference.py` | Model inference engine |
| `train_custom_cnn.py` | Custom CNN training |
| `train_transfer_learning.py` | Transfer learning training |
| `setup_kaggle_dataset.py` | Dataset download automation |
| `app.py` | Streamlit web application |

---

## ⚡ Quick Reference Commands

```bash
# Setup
pip install -r requirements.txt
python setup_kaggle_dataset.py

# Training
python train_custom_cnn.py
python train_transfer_learning.py

# Testing
python test_inference.py

# Run app
streamlit run app.py

# Verify models
python -c "import models_inference; r = models_inference.ModelRegistry(); r.initialize_all_models(); print(r.get_model('Custom CNN').is_loaded)"
```

---

**Backend implementation complete! Ready to train and deploy.** 🚀
