"""
================================================================================
                    DATASET_LOADER.PY
        Kaggle Fruits Dataset Loading & Preprocessing Utilities
================================================================================
Purpose:
    Handle dataset discovery, organization, and loading from the Kaggle
    Fruits Dataset (https://www.kaggle.com/datasets/shivamardeshna/fruits-dataset)
    Provides utilities for train/validation/test split and batch generation.
================================================================================
"""

import os
import shutil
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import config

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class FruitsDatasetLoader:
    """
    Manages loading and preprocessing the Kaggle Fruits Dataset.
    
    Expected dataset structure:
    fruits-dataset/
    ├── apple/
    │   ├── (images)
    ├── banana/
    │   ├── (images)
    └── orange/
        ├── (images)
    
    OR with fresh/rotten subdirectories:
    fruits-dataset/
    ├── Apple/
    │   ├── fresh/ → Fresh Apple (class 0)
    │   └── rotten/ → Rotten Apple (class 1)
    ├── Banana/
    │   ├── fresh/ → Fresh Banana (class 2)
    │   └── rotten/ → Rotten Banana (class 3)
    └── Orange/
        ├── fresh/ → Fresh Orange (class 4)
        └── rotten/ → Rotten Orange (class 5)
    """
    
    def __init__(self, dataset_path: Path = None):
        """
        Initialize dataset loader.
        
        Args:
            dataset_path: Path to fruits dataset root directory
        """
        if dataset_path is None:
            dataset_path = config.DATA_DIR / "fruits-dataset"
        
        self.dataset_path = Path(dataset_path)
        self.class_mapping = self._build_class_mapping()
        self.image_paths = []
        self.image_labels = []
    
    def _build_class_mapping(self) -> Dict[str, int]:
        """Build mapping from directory structure to class indices."""
        mapping = {}
        
        # Standard mapping: Fruit type × freshness
        mapping["Apple/fresh"] = 0
        mapping["Apple/rotten"] = 1
        mapping["Banana/fresh"] = 2
        mapping["Banana/rotten"] = 3
        mapping["Orange/fresh"] = 4
        mapping["Orange/rotten"] = 5
        
        # Alternative: lowercase/singular
        mapping["apple/fresh"] = 0
        mapping["apple/rotten"] = 1
        mapping["banana/fresh"] = 2
        mapping["banana/rotten"] = 3
        mapping["orange/fresh"] = 4
        mapping["orange/rotten"] = 5
        
        return mapping
    
    def _find_images_in_directory(self, directory: Path) -> List[Tuple[Path, int]]:
        """
        Recursively find all images and map to class labels.
        
        Returns:
            List of (image_path, class_idx) tuples
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        image_class_pairs = []
        
        if not directory.exists():
            print(f"⚠ Dataset directory not found: {directory}")
            return image_class_pairs
        
        # Traverse directory structure
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix in valid_extensions:
                    full_path = Path(root) / file
                    
                    # Determine class based on directory structure
                    class_idx = self._map_path_to_class(full_path)
                    if class_idx is not None:
                        image_class_pairs.append((full_path, class_idx))
        
        print(f"✓ Found {len(image_class_pairs)} images in {directory}")
        return image_class_pairs
    
    def _map_path_to_class(self, image_path: Path) -> int:
        """
        Determine class index from image file path.
        
        Args:
            image_path: Full path to image file
            
        Returns:
            Class index (0-5) or None if cannot determine
        """
        relative_path = image_path.relative_to(self.dataset_path)
        path_str = str(relative_path).replace("\\", "/")
        
        # Try direct mapping
        for pattern, class_idx in self.class_mapping.items():
            if pattern in path_str:
                return class_idx
        
        # Try partial matching (e.g., "apple" in path → Apple class)
        path_lower = path_str.lower()
        for pattern, class_idx in self.class_mapping.items():
            pattern_lower = pattern.lower()
            if pattern_lower in path_lower:
                return class_idx
        
        return None
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all images from dataset.
        
        Returns:
            Tuple of:
                - Image array (N, H, W, 3) in uint8 format
                - Label array (N,)
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {self.dataset_path}\n"
                f"Please download dataset from:\n"
                f"https://www.kaggle.com/datasets/shivamardeshna/fruits-dataset"
            )
        
        image_class_pairs = self._find_images_in_directory(self.dataset_path)
        
        if len(image_class_pairs) == 0:
            raise ValueError("No images found in dataset directory")
        
        images = []
        labels = []
        
        print(f"\nLoading {len(image_class_pairs)} images...")
        for idx, (image_path, class_idx) in enumerate(image_class_pairs):
            if (idx + 1) % max(1, len(image_class_pairs) // 10) == 0:
                print(f"  Progress: {idx+1}/{len(image_class_pairs)}")
            
            try:
                # Load image
                img = load_img(
                    image_path,
                    target_size=(config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH)
                )
                img_array = img_to_array(img).astype(np.uint8)
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"⚠ Failed to load {image_path}: {e}")
                continue
        
        self.image_paths = [p for p, _ in image_class_pairs[:len(images)]]
        self.image_labels = labels
        
        images_array = np.array(images)
        labels_array = np.array(labels)
        
        print(f"✓ Loaded {len(images)} images successfully")
        print(f"  Shape: {images_array.shape}")
        print(f"  Class distribution: {np.bincount(labels_array)}")
        
        return images_array, labels_array
    
    def create_train_val_test_split(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Split dataset into train/validation/test sets, stratified by class.
        
        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        np.random.seed(random_seed)
        
        # Stratified split by class
        indices = np.arange(len(labels))
        n_samples = len(labels)
        
        # Get indices per class
        split_indices = []
        for class_idx in np.unique(labels):
            class_indices = np.where(labels == class_idx)[0]
            np.random.shuffle(class_indices)
            
            # Split this class's samples
            n_class = len(class_indices)
            n_train = int(n_class * train_ratio)
            n_val = int(n_class * val_ratio)
            
            train_idx = class_indices[:n_train]
            val_idx = class_indices[n_train:n_train+n_val]
            test_idx = class_indices[n_train+n_val:]
            
            split_indices.append({
                'train': train_idx,
                'val': val_idx,
                'test': test_idx,
            })
        
        # Aggregate indices
        train_idx = np.concatenate([s['train'] for s in split_indices])
        val_idx = np.concatenate([s['val'] for s in split_indices])
        test_idx = np.concatenate([s['test'] for s in split_indices])
        
        # Shuffle
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)
        
        # Create splits
        train_set = (images[train_idx], labels[train_idx])
        val_set = (images[val_idx], labels[val_idx])
        test_set = (images[test_idx], labels[test_idx])
        
        print(f"\n✓ Dataset split:")
        print(f"  Training:   {len(train_idx)} samples ({len(train_idx)/n_samples*100:.1f}%)")
        print(f"  Validation: {len(val_idx)} samples ({len(val_idx)/n_samples*100:.1f}%)")
        print(f"  Test:       {len(test_idx)} samples ({len(test_idx)/n_samples*100:.1f}%)")
        
        return train_set, val_set, test_set
    
    def create_data_generators(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
    ):
        """
        Create ImageDataGenerator for training with augmentation.
        
        Returns:
            Tuple of (train_generator, preprocessing_function)
        """
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            rescale=1./255
        )
        
        # Convert labels to one-hot
        train_labels_onehot = tf.keras.utils.to_categorical(
            train_labels,
            num_classes=config.NUM_CLASSES
        )
        
        train_generator = train_datagen.flow(
            train_images,
            train_labels_onehot,
            batch_size=config.MAX_BATCH_SIZE,
            shuffle=True
        )
        
        return train_generator
    
    @staticmethod
    def get_dataset_info() -> Dict:
        """Get information about dataset statistics."""
        return {
            "name": "Kaggle Fruits Dataset",
            "url": "https://www.kaggle.com/datasets/shivamardeshna/fruits-dataset",
            "classes": list(config.FRUIT_CLASSES.values()),
            "num_classes": config.NUM_CLASSES,
            "image_size": (config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH),
            "total_images_expected": "~11,000",
            "fruit_types": list(config.FRUIT_TYPES.keys()),
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def setup_dataset_directory(force_recreate: bool = False) -> Path:
    """
    Ensure dataset directory exists and is organized.
    
    Args:
        force_recreate: If True, remove and recreate directory
        
    Returns:
        Path to dataset directory
    """
    dataset_dir = config.DATA_DIR / "fruits-dataset"
    
    if force_recreate and dataset_dir.exists():
        print(f"Removing existing directory: {dataset_dir}")
        shutil.rmtree(dataset_dir)
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Dataset directory ready: {dataset_dir}")
    
    return dataset_dir


def print_dataset_info():
    """Print dataset information and download instructions."""
    info = FruitsDatasetLoader.get_dataset_info()
    
    print("\n" + "="*70)
    print("KAGGLE FRUITS DATASET INFORMATION")
    print("="*70)
    print(f"Name: {info['name']}")
    print(f"URL: {info['url']}")
    print(f"Classes: {len(info['classes'])}")
    for i, class_name in enumerate(info['classes']):
        print(f"  {i}: {class_name}")
    print(f"Image Size: {info['image_size']}")
    print(f"Expected Total Images: {info['total_images_expected']}")
    print("="*70 + "\n")
