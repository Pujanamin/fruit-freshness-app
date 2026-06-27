# 🚀 Quick Start Guide - Complete Workflow

## 5-Minute Setup (Without Pre-trained Models)

### 1. Install Dependencies
```bash
cd fruit-freshness-app
pip install -r requirements.txt
```

### 2. Run Web Application
```bash
streamlit run app.py
```

### 3. Open Browser
Navigate to `http://localhost:8501`

✅ **Done!** The app works with fallback inference (no models needed)

---

## 30-Minute Setup (Full Backend Training)

### Prerequisites
- ✅ Python 3.9+
- ✅ 10GB disk space
- ✅ Kaggle account (free)
- ✅ 30-60 minutes training time (depends on CPU/GPU)

### Step 1: Install & Setup Dataset (10 min)

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle (automatic)
python setup_kaggle_dataset.py
```

**If Kaggle API setup fails:**
- Manual download: https://www.kaggle.com/datasets/shivamardeshna/fruits-dataset
- Extract to: `data/fruits-dataset/`
- Verify: `ls data/fruits-dataset/Apple/fresh/` (should show images)

### Step 2: Train Custom CNN (15-20 min)

```bash
python train_custom_cnn.py
```

**Progress:**
- Loading dataset...
- Building model...
- Training [████████░░] 80% accuracy

**Output:** `models/custom_cnn_fruit_classifier.h5`

### Step 3: Train Transfer Learning (15-25 min)

```bash
python train_transfer_learning.py
```

**Progress:**
- Stage 1: Feature extraction (frozen backbone)
- Stage 2: Fine-tuning (unfrozen backbone)

**Output:** `models/transfer_learning_fruit_classifier.h5`

### Step 4: Launch Web App

```bash
streamlit run app.py
```

✅ **Complete!** Both models loaded and ready

---

## Testing the System

### Test 1: Dashboard
1. Open app in browser
2. Click "Dashboard" tab
3. View research questions & benchmarks
4. ✅ Verify metrics displayed

### Test 2: Screening Tool
1. Click "Screening Tool" tab
2. Upload a fruit image (JPG/PNG)
3. View prediction & confidence
4. View Grad-CAM heatmap
5. ✅ Export results as JSON/CSV

### Test 3: Failure Analysis
1. Click "Failure Analysis" tab
2. Review operational constraints
3. Read known failure patterns
4. Understand human-in-the-loop workflow
5. ✅ Review deployment checklist

---

## GPU Acceleration (Optional - 5x Faster)

### Prerequisites
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed

### Install GPU Support
```bash
pip uninstall tensorflow -y
pip install tensorflow-gpu
```

### Verify GPU
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Training Speedup
- CPU: 60-90 min total
- GPU: 15-20 min total

---

## Directory Structure Check

Verify everything is in place:

```
fruit-freshness-app/
├── app.py                           # ✅ Web application
├── config.py                        # ✅ Configuration
├── utils.py                         # ✅ Image utilities
├── models_inference.py              # ✅ Inference engine
├── dataset_loader.py                # ✅ Dataset utilities (NEW)
├── train_custom_cnn.py              # ✅ CNN training (NEW)
├── train_transfer_learning.py       # ✅ Transfer learning (NEW)
├── setup_kaggle_dataset.py          # ✅ Dataset setup (NEW)
├── requirements.txt                 # ✅ Updated dependencies
├── models/                          # ✅ Directory created
│   ├── custom_cnn_fruit_classifier.h5
│   └── transfer_learning_fruit_classifier.h5
└── data/                            # ✅ Directory created
    └── fruits-dataset/             # ✅ Dataset downloaded
        ├── Apple/
        ├── Banana/
        └── Orange/
```

---

## Common Commands Reference

```bash
# Setup
python setup_kaggle_dataset.py          # Download & verify dataset
python setup_kaggle_dataset.py --verify-only  # Check existing dataset

# Training
python train_custom_cnn.py              # Train Custom CNN (15-20 min)
python train_transfer_learning.py       # Train Transfer Learning (20-25 min)

# Testing
python -c "import app; import models_inference; print('✅ All imports successful')"

# Web App
streamlit run app.py                    # Launch application
streamlit run app.py --server.port 8502 # Launch on different port
```

---

## Troubleshooting Quick Fixes

| Issue | Fix |
|-------|-----|
| "No module named tensorflow" | `pip install tensorflow` |
| "Kaggle API not found" | Setup: https://www.kaggle.com/settings/account |
| "Dataset not found" | Run: `python setup_kaggle_dataset.py` |
| Port 8501 already in use | Use: `streamlit run app.py --server.port 8502` |
| Training very slow | Install GPU: `pip install tensorflow-gpu` |
| Out of memory | Reduce: `MAX_BATCH_SIZE = 16` in config.py |

---

## Expected Performance

### Quick Test (No Training)
- App loads: ✅ Instant
- Fallback inference: ✅ <2 seconds
- Grad-CAM heatmap: ✅ <1 second
- **Perfect for demos!**

### After Training
- Custom CNN inference: ✅ 0.3-0.5 seconds
- Transfer Learning inference: ✅ 0.5-0.8 seconds
- Test Accuracy (Custom CNN): ✅ ~87-89%
- Test Accuracy (Transfer Learning): ✅ ~90-92%

---

## Next Steps

1. **For Development:**
   - Review `BACKEND_GUIDE.md` for detailed training
   - Edit `config.py` to customize settings
   - Add custom preprocessing in `utils.py`

2. **For Production:**
   - Read `README.md` for architecture overview
   - Review `SETUP_GUIDE.md` for deployment
   - Implement logging and monitoring

3. **For Research:**
   - Analyze training metrics in `models/metrics_*.json`
   - Compare model performance
   - Fine-tune hyperparameters

---

## Support Resources

- 📖 **README.md** - Comprehensive guide
- 🔧 **BACKEND_GUIDE.md** - Detailed backend setup
- ⚙️ **SETUP_GUIDE.md** - Installation walkthrough
- 📋 **FILE_MANIFEST.md** - File reference

---

**Ready? Start with:** `pip install -r requirements.txt && streamlit run app.py` 🎉
