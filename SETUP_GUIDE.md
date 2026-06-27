"""
================================================================================
                      SETUP_GUIDE.md
         Step-by-Step Installation and Configuration Guide
================================================================================
"""

# 🚀 Fruit Freshness Inspector - Complete Setup Guide

## Prerequisites Check

Before starting, ensure you have:
- [ ] **Python 3.9 or higher** - Check with: `python --version`
- [ ] **pip package manager** - Check with: `pip --version`
- [ ] **Git** (optional, for version control)
- [ ] **10GB free disk space** (for dependencies + models)
- [ ] **Modern web browser** (Chrome, Firefox, Safari, Edge)

---

## Step 1: Create Project Directory

```bash
# Navigate to desired location
cd /path/to/your/projects

# Create project folder
mkdir fruit-freshness-app
cd fruit-freshness-app

# Initialize git (optional)
git init
```

---

## Step 2: Create Python Virtual Environment

**Why?** Virtual environment isolates project dependencies from system Python.

### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### On Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### On Windows (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Verify activation:**
- Prompt should show `(venv)` prefix
- Command: `which python` (Mac/Linux) or `where python` (Windows)

---

## Step 3: Install Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import streamlit; import tensorflow; import cv2; print('✓ All imports successful')"
```

**Installation typically takes 5-10 minutes** depending on internet speed.

---

## Step 4: Organize Directory Structure

Create required subdirectories:

```bash
# Create directories
mkdir models
mkdir data
mkdir uploads

# Create .gitignore (optional)
echo "uploads/*" > .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".DS_Store" >> .gitignore
```

---

## Step 5: Prepare Model Files (Optional)

If you have pre-trained H5 models:

```
1. Place in models/ directory:
   models/custom_cnn_fruit_classifier.h5
   models/transfer_learning_fruit_classifier.h5

2. Models should be:
   - Format: TensorFlow/Keras .h5
   - Input shape: (224, 224, 3)
   - Output shape: (6,) for 6 classes
```

**Without models?** System operates in **fallback mode** using deterministic inference—perfect for testing!

---

## Step 6: Launch Application

```bash
streamlit run app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

✅ Open browser to `http://localhost:8501`

---

## Step 7: Test the Application

### **Quick Test Workflow:**

1. **Navigate to "Screening Tool"** in left sidebar
2. **Upload a test image** (fruit photo, any format: PNG/JPG/JPEG)
3. **Observe results:**
   - Predicted freshness status
   - Confidence score
   - Grad-CAM heatmap overlay
   - Export options

### **Dashboard Review:**
- Click "Dashboard" to see model benchmarks
- Review research questions
- Check dataset statistics

### **Failure Analysis:**
- Read known failure patterns
- Review deployment constraints
- Understand human-in-the-loop workflow

---

## Configuration Customization

### **Adjust Confidence Thresholds**

Edit `config.py`:

```python
# Stricter quality control: require 75%+ confidence
CONFIDENCE_THRESHOLD_WARNING = 0.75  # Flag for manual review
CONFIDENCE_THRESHOLD_REJECT = 0.55   # Auto-reject threshold

# Lenient for high-volume screening: 55%+ confidence
CONFIDENCE_THRESHOLD_WARNING = 0.55
CONFIDENCE_THRESHOLD_REJECT = 0.35
```

### **Change Input Image Size** (requires model retraining)

```python
IMAGE_TARGET_HEIGHT = 256  # Increase for higher resolution
IMAGE_TARGET_WIDTH = 256
```

### **Add Fruit Types**

```python
FRUIT_CLASSES = {
    0: "Fresh Apple",
    1: "Rotten Apple",
    2: "Fresh Banana",
    3: "Rotten Banana",
    4: "Fresh Orange",
    5: "Rotten Orange",
    6: "Fresh Mango",      # NEW
    7: "Rotten Mango",     # NEW
}

NUM_CLASSES = 8  # Update this too
```

---

## Troubleshooting

### **"ModuleNotFoundError: No module named 'streamlit'"**

```bash
# Reinstall all requirements
pip install -r requirements.txt --force-reinstall

# Or install specific package
pip install streamlit==1.28.1
```

### **Port 8501 Already in Use**

```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill process using port (macOS/Linux)
lsof -i :8501 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

### **"Model file not found" Warning**

This is **expected and normal**. System automatically uses fallback inference.
To use real models: place .h5 files in `models/` directory.

### **Slow Model Loading**

First inference is slower due to lazy-loading. Subsequent predictions are faster.

### **TensorFlow GPU Not Using GPU**

```bash
# Verify GPU available
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, TensorFlow using CPU (still works fine, just slower)
# For GPU support, reinstall: pip install tensorflow-gpu
```

---

## Deployment Scenarios

### **Scenario 1: Local Development & Testing**

```bash
# Default setup
streamlit run app.py
```

### **Scenario 2: Network Deployment (Multi-user)**

```bash
# Allow external access
streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.enableXsrfProtection false
```

**Access from:** `http://<your-ip>:8501`

### **Scenario 3: Production Deployment (Docker)**

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t fruit-inspector .
docker run -p 8501:8501 fruit-inspector
```

---

## Performance Optimization

### **For High Throughput (Batch Processing)**

Edit `config.py`:
```python
MAX_BATCH_SIZE = 32  # Process 32 images at once
```

### **For Low-Latency (Real-time)**

```python
# Reduce image size (if models support)
IMAGE_TARGET_HEIGHT = 128
IMAGE_TARGET_WIDTH = 128
```

### **GPU Acceleration**

```bash
# Install GPU-accelerated TensorFlow
pip uninstall tensorflow -y
pip install tensorflow-gpu
```

---

## Security Notes

⚠️ **Before Production Deployment:**

1. **API Authentication** - Add user login if exposed publicly
2. **File Upload Validation** - Restrict file types and sizes
3. **Data Privacy** - Implement image deletion after processing
4. **Audit Logging** - Log all predictions for compliance
5. **HTTPS** - Use SSL/TLS for network access

---

## Monitoring & Maintenance

### **Weekly Checks**
- Review prediction accuracy metrics
- Check for systematic failures
- Monitor confidence score distribution

### **Monthly Tasks**
- Analyze failure patterns
- Retrain model if needed
- Update documentation
- Backup prediction logs

### **Quarterly Reviews**
- Performance benchmarking
- Model update cycle
- Operator feedback review
- System optimization

---

## Getting Sample Data

### **Option 1: Use Public Fruit Datasets**
- Kaggle: "Fruit Fresh and Rotten for Classification"
- Google Images: Search "apple fresh rotten"
- UC Davis Agricultural Data: Farm produce datasets

### **Option 2: Capture Your Own**
- Use smartphone camera
- Uniform lighting setup
- Single fruit per image
- Save as PNG or JPG

### **Option 3: Synthetic Demo Images**
Already supported! System works without trained models.

---

## Next Steps

1. **✓ Installation Complete** - System is ready to use
2. **→ Upload test image** - Try Screening Tool
3. **→ Review benchmarks** - Check Dashboard for metrics
4. **→ Understand failures** - Read Failure Analysis page
5. **→ Deploy to production** - Follow deployment guidelines

---

## Support Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **TensorFlow/Keras**: https://www.tensorflow.org
- **OpenCV Docs**: https://docs.opencv.org
- **Project README**: See `README.md` for comprehensive guide

---

## Quick Commands Reference

```bash
# Activate environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Start application
streamlit run app.py

# Install dependencies
pip install -r requirements.txt

# Upgrade pip
pip install --upgrade pip

# Deactivate environment
deactivate

# Run tests (if available)
pytest tests/

# Check Python version
python --version
```

---

**Ready to inspect fruit freshness? 🍎 Let's go!**
