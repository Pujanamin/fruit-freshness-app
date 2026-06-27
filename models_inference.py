"""
================================================================================
                        MODELS_INFERENCE.PY
    Lazy Model Loading, Inference Pipeline & Deterministic Fallback Logic
================================================================================
Purpose:
    Manages lazy loading of TensorFlow H5 model files with robust error handling.
    Implements multi-class prediction logic and deterministic string-matching 
    fallback inference when local weights are unavailable. Provides synthetic 
    benchmark predictions to enable seamless system demonstration without 
    requiring pre-trained model files.
================================================================================
"""

import os
import numpy as np
from typing import Tuple, Optional, Dict
import config

# Import TensorFlow with graceful degradation if unavailable
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Using fallback inference mode.")


# ============================================================================
# MODEL LOADER WITH LAZY INITIALIZATION
# ============================================================================
class FruitClassifierModel:
    """
    Lazy-loading wrapper for Keras H5 models with fallback inference support.
    Loads models on first inference call, caching in memory for performance.
    """
    
    def __init__(self, model_path: str, model_name: str):
        """
        Initialize model wrapper.
        
        Args:
            model_path: Path to H5 model file
            model_name: Human-readable model identifier
        """
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        self.load_error = None
        self.inference_mode = "tensorflow"  # or "fallback"
    
    def _load_model(self) -> bool:
        """
        Attempt to load Keras model from H5 file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not TENSORFLOW_AVAILABLE:
            self.inference_mode = "fallback"
            self.load_error = "TensorFlow not available"
            return False
        
        if not os.path.exists(self.model_path):
            self.inference_mode = "fallback"
            self.load_error = f"Model file not found: {self.model_path}"
            return False
        
        try:
            # Suppress TensorFlow logging for cleaner UI
            tf.get_logger().setLevel('ERROR')
            self.model = keras.models.load_model(self.model_path)
            self.is_loaded = True
            self.inference_mode = "tensorflow"
            print(f"✓ Model loaded: {self.model_name}")
            return True
        except Exception as e:
            self.load_error = str(e)
            self.inference_mode = "fallback"
            print(f"⚠ Model load failed: {self.load_error}. Using fallback mode.")
            return False
    
    def predict(self, image_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on image batch using loaded model or fallback logic.
        
        Args:
            image_batch: Preprocessed image tensor (N, H, W, 3)
            
        Returns:
            Tuple of:
                - Predicted class indices (N,)
                - Confidence scores (N, num_classes)
        """
        # Lazy load on first call
        if not self.is_loaded and self.inference_mode != "fallback":
            self._load_model()
        
        # Use loaded model if available
        if self.is_loaded and self.model is not None:
            try:
                logits = self.model.predict(image_batch, verbose=0)
                predictions = np.argmax(logits, axis=1)
                confidences = tf.nn.softmax(logits).numpy()
                return predictions, confidences
            except Exception as e:
                print(f"Inference error: {e}. Switching to fallback mode.")
                self.inference_mode = "fallback"
        
        # Fallback: deterministic synthetic prediction
        return self._fallback_predict(image_batch)
    
    def _fallback_predict(self, image_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ENHANCED fallback inference: HSV color + Laplacian texture analysis.
        Identifies fruit type from hue, freshness from saturation+brightness+texture.
        
        Args:
            image_batch: Preprocessed ImageNet-normalized tensor (N, H, W, 3)
            
        Returns:
            Tuple of (predictions[N], confidences[N, 6])
        """
        import cv2
        batch_size = image_batch.shape[0]
        predictions = np.zeros(batch_size, dtype=np.int32)
        confidences = np.zeros((batch_size, config.NUM_CLASSES))
        
        for idx, image in enumerate(image_batch):
            # Reverse ImageNet normalization
            mean_array = np.array(config.IMAGE_NORMALIZE_MEAN).reshape(1, 1, 3)
            std_array = np.array(config.IMAGE_NORMALIZE_STD).reshape(1, 1, 3)
            image_float = image * std_array + mean_array
            image_uint8 = np.clip(image_float * 255, 0, 255).astype(np.uint8)
            
            # Convert to HSV
            if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
                hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
                hue_channel = hsv[:, :, 0]
                saturation_channel = hsv[:, :, 1]
                value_channel = hsv[:, :, 2]
                
                # Histogram-based hue (robust to lighting)
                hist_hue = cv2.calcHist([hue_channel.astype(np.uint8)], [0], None, [180], [0, 180])
                dominant_hue = float(np.argmax(hist_hue)) if hist_hue.max() > 0 else 90
                
                mean_saturation = np.mean(saturation_channel)
                mean_value = np.mean(value_channel)
                hue_std = np.std(hue_channel)  # Color consistency
                
                # Texture via Laplacian edge detection
                gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                texture_roughness = np.mean(np.abs(laplacian))
                
                saturated_pixels = np.mean(saturation_channel > 100)
            else:
                dominant_hue = 90
                mean_saturation = 128
                mean_value = 128
                hue_std = 30
                texture_roughness = 0
                saturated_pixels = 0.5
            
            # FRUIT TYPE CLASSIFICATION by hue
            if dominant_hue < 15 or dominant_hue > 165:
                fruit_type = 0  # Apple (RED)
                fruit_confidence = 0.92
            elif 35 <= dominant_hue <= 85:
                fruit_type = 1  # Banana (YELLOW/GREEN)
                fruit_confidence = 0.90
            elif 10 <= dominant_hue < 35:
                fruit_type = 2  # Orange (ORANGE)
                fruit_confidence = 0.88
            else:
                # Ambiguous: use saturation to break tie
                if saturated_pixels > 0.40:
                    fruit_type = 1  # Banana
                    fruit_confidence = 0.72
                else:
                    fruit_type = 2  # Orange
                    fruit_confidence = 0.72
            
            # FRESHNESS DETECTION: Combine 4 factors
            sat_score = np.clip(mean_saturation / 200.0, 0, 1)  # Vibrant = fresh
            bright_score = np.clip(mean_value / 220.0, 0, 1)  # Bright = fresh
            consistency_score = np.clip(1.0 - (hue_std / 50.0), 0, 1)  # Uniform color = fresh
            
            # Texture interpretation (depends on fruit type)
            if fruit_type == 0:  # Apple: smooth when fresh
                texture_score = 1.0 - np.clip(texture_roughness / 45.0, 0, 1)
            elif fruit_type == 1:  # Banana: smooth when fresh
                texture_score = 1.0 - np.clip(texture_roughness / 35.0, 0, 1)
            else:  # Orange: naturally bumpy, but not too rough
                texture_score = 1.0 - np.clip(texture_roughness / 55.0, 0, 1)
            
            # Weighted freshness
            freshness_score = (
                0.40 * sat_score +
                0.32 * bright_score +
                0.18 * consistency_score +
                0.10 * texture_score
            )
            freshness_score = np.clip(freshness_score, 0, 1)
            
            # Rotten threshold
            is_rotten = freshness_score < 0.35
            
            # Map to class
            class_idx = fruit_type * 2 + (1 if is_rotten else 0)
            predictions[idx] = class_idx
            
            # CONFIDENCE: modified by freshness certainty
            if is_rotten:
                confidence_base = 0.52 + (1.0 - freshness_score) * 0.38
            else:
                confidence_base = 0.56 + freshness_score * 0.37
            
            confidence_base = np.clip(confidence_base * fruit_confidence, 0.45, 0.96)
            confidences[idx, class_idx] = confidence_base
            
            # Distribute remaining probability
            remaining_prob = 1.0 - confidence_base
            for i in range(config.NUM_CLASSES):
                if i != class_idx:
                    confidences[idx, i] = remaining_prob / (config.NUM_CLASSES - 1)
        
        return predictions, confidences


# ============================================================================
# GLOBAL MODEL REGISTRY & LAZY INITIALIZATION
# ============================================================================
class ModelRegistry:
    """Singleton registry for managing multiple model instances."""
    
    _instance = None
    _models: Dict[str, FruitClassifierModel] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register_model(self, model_name: str, model_path: str) -> None:
        """Register a model for lazy loading."""
        if model_name not in self._models:
            self._models[model_name] = FruitClassifierModel(model_path, model_name)
    
    def get_model(self, model_name: str) -> Optional[FruitClassifierModel]:
        """Retrieve registered model by name."""
        return self._models.get(model_name)
    
    def initialize_all_models(self) -> None:
        """Pre-register all configured models."""
        for model_name, model_path in config.AVAILABLE_MODELS.items():
            self.register_model(model_name, str(model_path))


# ============================================================================
# INFERENCE PIPELINE
# ============================================================================
def run_inference(
    image_batch: np.ndarray,
    model_name: str = "Custom CNN",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute inference pipeline with automatic model selection and fallback.
    
    Args:
        image_batch: Preprocessed image tensor (N, H, W, 3)
        model_name: Name of model to use
        
    Returns:
        Tuple of:
            - Predicted class indices (N,)
            - Confidence scores (N, num_classes)
    """
    registry = ModelRegistry()
    model = registry.get_model(model_name)
    
    if model is None:
        raise ValueError(f"Model '{model_name}' not registered")
    
    predictions, confidences = model.predict(image_batch)
    return predictions, confidences


def get_inference_metadata(model_name: str) -> Dict:
    """
    Get metadata about current inference mode and model status.
    
    Returns:
        dict: Inference mode, load status, error messages if any
    """
    registry = ModelRegistry()
    model = registry.get_model(model_name)
    
    if model is None:
        return {"status": "not_registered", "error": "Model not found"}
    
    return {
        "model_name": model.model_name,
        "inference_mode": model.inference_mode,
        "is_loaded": model.is_loaded,
        "load_error": model.load_error,
        "model_path": model.model_path,
    }


# ============================================================================
# POST-PROCESSING & RESULT FORMATTING
# ============================================================================
def format_prediction_result(
    class_idx: int,
    confidence_scores: np.ndarray,
) -> Dict:
    """
    Format raw prediction into human-readable result dictionary.
    
    Args:
        class_idx: Predicted class index
        confidence_scores: Confidence vector (num_classes,)
        
    Returns:
        dict: Formatted prediction with class names, confidence, freshness status
    """
    class_name = config.FRUIT_CLASSES.get(class_idx, "Unknown")
    freshness = config.FRUIT_FRESHNESS_MAPPING.get(class_idx, "Unknown")
    confidence = float(confidence_scores[class_idx])
    
    # Extract fruit type
    fruit_type = None
    for fruit, class_range in config.FRUIT_TYPES.items():
        if class_idx in class_range:
            fruit_type = fruit
            break
    
    # Calculate per-class confidence breakdown
    class_confidences = {
        config.FRUIT_CLASSES[i]: float(confidence_scores[i])
        for i in range(config.NUM_CLASSES)
    }
    
    return {
        "predicted_class": class_name,
        "fruit_type": fruit_type,
        "freshness_status": freshness,
        "confidence": confidence,
        "class_confidences": class_confidences,
        "requires_manual_review": confidence < config.CONFIDENCE_THRESHOLD_WARNING,
        "auto_reject": confidence < config.CONFIDENCE_THRESHOLD_REJECT,
    }


def get_benchmark_comparison(model_name: str, metric_name: str) -> Optional[float]:
    """
    Retrieve benchmark metric for model comparison in dashboards.
    
    Args:
        model_name: Model identifier
        metric_name: Metric to retrieve
        
    Returns:
        float: Metric value or None if not available
    """
    return config.BENCHMARK_METRICS.get(model_name, {}).get(metric_name)


# ============================================================================
# INITIALIZATION
# ============================================================================
# Initialize model registry on module import
_registry = ModelRegistry()
_registry.initialize_all_models()
