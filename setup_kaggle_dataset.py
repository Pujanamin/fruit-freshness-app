"""
================================================================================
                    SETUP_KAGGLE_DATASET.PY
        Automated Kaggle Dataset Download & Organization Script
================================================================================
Purpose:
    Automate downloading the Fruits Dataset from Kaggle and organizing it
    into the correct directory structure for training.

Requirements:
    - Kaggle API credentials (~/.kaggle/kaggle.json)
    - Download instructions: https://www.kaggle.com/settings/account

Usage:
    python setup_kaggle_dataset.py

This script will:
    1. Verify Kaggle API is configured
    2. Download fruits-dataset from Kaggle
    3. Extract and organize files
    4. Verify dataset structure
    5. Display dataset statistics
================================================================================
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict

import config
from dataset_loader import FruitsDatasetLoader, print_dataset_info


def check_kaggle_api() -> bool:
    """
    Check if Kaggle API is installed and configured.
    
    Returns:
        bool: True if Kaggle API is available and credentials found
    """
    try:
        import kaggle
        
        # Check for credentials
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        
        if not kaggle_json.exists():
            print("⚠ Kaggle API credentials not found")
            print(f"Expected location: {kaggle_json}")
            print("\nTo set up Kaggle API:")
            print("1. Go to: https://www.kaggle.com/settings/account")
            print("2. Click 'Create New API Token'")
            print("3. Save kaggle.json to ~/.kaggle/")
            print("4. Run: chmod 600 ~/.kaggle/kaggle.json (on Mac/Linux)")
            return False
        
        print("✓ Kaggle API credentials found")
        return True
    except ImportError:
        print("⚠ kaggle-api not installed")
        print("Install with: pip install kaggle")
        return False


def download_kaggle_dataset(dataset_name: str = "shivamardeshna/fruits-dataset") -> bool:
    """
    Download dataset from Kaggle API.
    
    Args:
        dataset_name: Kaggle dataset identifier
        
    Returns:
        bool: True if download successful
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print(f"\n[1/3] Downloading dataset: {dataset_name}")
        print(f"       Destination: {config.DATA_DIR}")
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        api.dataset_download_files(
            dataset_name,
            path=str(config.DATA_DIR),
            unzip=True
        )
        
        print("✓ Dataset downloaded and extracted")
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def organize_dataset_structure() -> bool:
    """
    Organize downloaded dataset into expected directory structure.
    
    Returns:
        bool: True if organization successful
    """
    print("\n[2/3] Organizing dataset structure...")
    
    dataset_dir = config.DATA_DIR / "fruits-dataset"
    
    if not dataset_dir.exists():
        print(f"⚠ Dataset directory not found: {dataset_dir}")
        return False
    
    # Check various possible directory structures
    potential_subdirs = [
        "fruits-dataset",
        "Fruits-Dataset",
        "fruits",
        "data",
    ]
    
    for subdir in potential_subdirs:
        possible_path = config.DATA_DIR / subdir
        if possible_path.exists() and possible_path.is_dir():
            if possible_path != dataset_dir:
                print(f"  Found: {possible_path}")
                # Move to standard location if different
                if dataset_dir.exists():
                    shutil.rmtree(dataset_dir)
                shutil.move(str(possible_path), str(dataset_dir))
                print(f"  Moved to: {dataset_dir}")
    
    # Create expected subdirectories if needed
    fruit_types = ["Apple", "Banana", "Orange"]
    freshness_types = ["fresh", "rotten"]
    
    for fruit in fruit_types:
        fruit_dir = dataset_dir / fruit
        if not fruit_dir.exists():
            print(f"  Creating: {fruit_dir}")
            fruit_dir.mkdir(parents=True, exist_ok=True)
            
            for freshness in freshness_types:
                freshness_dir = fruit_dir / freshness
                freshness_dir.mkdir(parents=True, exist_ok=True)
    
    print("✓ Dataset structure organized")
    return True


def verify_dataset_integrity() -> Dict:
    """
    Verify dataset has expected structure and content.
    
    Returns:
        dict: Verification results
    """
    print("\n[3/3] Verifying dataset integrity...")
    
    dataset_dir = config.DATA_DIR / "fruits-dataset"
    stats = {
        "exists": dataset_dir.exists(),
        "images_found": 0,
        "classes": {},
        "valid": False,
    }
    
    if not stats["exists"]:
        print(f"✗ Dataset directory not found: {dataset_dir}")
        return stats
    
    # Count images per class
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if Path(file).suffix in image_extensions:
                stats["images_found"] += 1
                
                # Determine class
                relative_path = Path(root).relative_to(dataset_dir)
                class_key = str(relative_path)
                
                if class_key not in stats["classes"]:
                    stats["classes"][class_key] = 0
                stats["classes"][class_key] += 1
    
    # Verify minimum expected images
    stats["valid"] = stats["images_found"] > 1000  # Expect at least 1000 images
    
    print(f"✓ Found {stats['images_found']} images")
    print(f"  Classes: {len(stats['classes'])}")
    for class_name, count in sorted(stats["classes"].items()):
        print(f"    {class_name}: {count} images")
    
    if stats["valid"]:
        print("\n✓ Dataset verification PASSED")
    else:
        print("\n⚠ Dataset verification WARNING")
        print("  Expected at least 1000 images")
    
    return stats


def setup_complete_workflow():
    """Complete setup workflow with all steps."""
    
    print("\n" + "="*70)
    print("KAGGLE FRUITS DATASET - SETUP WORKFLOW")
    print("="*70)
    
    print_dataset_info()
    
    # Step 1: Check Kaggle API
    if not check_kaggle_api():
        print("\n⚠ Cannot proceed without Kaggle API setup")
        print("Please configure Kaggle API credentials and try again")
        return False
    
    # Step 2: Download dataset
    if not download_kaggle_dataset():
        print("\n⚠ Dataset download failed")
        print("Please try again or download manually from:")
        print("https://www.kaggle.com/datasets/shivamardeshna/fruits-dataset")
        return False
    
    # Step 3: Organize structure
    if not organize_dataset_structure():
        print("\n⚠ Dataset organization failed")
        return False
    
    # Step 4: Verify integrity
    stats = verify_dataset_integrity()
    
    if not stats["valid"]:
        print("\n⚠ Dataset verification failed")
        print("The dataset may be incomplete or corrupted")
        return False
    
    # Summary
    print("\n" + "="*70)
    print("✓ SETUP COMPLETE")
    print("="*70)
    print(f"\nDataset Location: {config.DATA_DIR / 'fruits-dataset'}")
    print(f"Total Images: {stats['images_found']}")
    print(f"Classes: {len(stats['classes'])}")
    print("\nNext steps:")
    print("  1. Train Custom CNN:        python train_custom_cnn.py")
    print("  2. Train Transfer Learning: python train_transfer_learning.py")
    print("  3. Run Web Application:     streamlit run app.py")
    
    return True


def manual_setup_instructions():
    """Print manual setup instructions if automated download fails."""
    print("\n" + "="*70)
    print("MANUAL DATASET SETUP INSTRUCTIONS")
    print("="*70)
    
    print("""
1. Visit: https://www.kaggle.com/datasets/shivamardeshna/fruits-dataset

2. Click "Download" button

3. Extract the downloaded ZIP file to:
   """)
    print(f"   {config.DATA_DIR}/fruits-dataset")
    
    print("""
4. Organize the directory structure:
   fruits-dataset/
   ├── Apple/
   │   ├── fresh/
   │   │   └── (apple images)
   │   └── rotten/
   │       └── (apple images)
   ├── Banana/
   │   ├── fresh/
   │   │   └── (banana images)
   │   └── rotten/
   │       └── (banana images)
   └── Orange/
       ├── fresh/
       │   └── (orange images)
       └── rotten/
           └── (orange images)

5. Verify the structure with:
   python setup_kaggle_dataset.py --verify-only

6. Train models:
   python train_custom_cnn.py
   python train_transfer_learning.py
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Kaggle Fruits Dataset")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing dataset (skip download)"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Print manual setup instructions"
    )
    
    args = parser.parse_args()
    
    if args.manual:
        manual_setup_instructions()
    elif args.verify_only:
        print("Verifying existing dataset...")
        stats = verify_dataset_integrity()
        if stats["valid"]:
            print("\n✓ Dataset is valid and ready for training")
        else:
            print("\n⚠ Dataset needs to be set up")
            print("Run: python setup_kaggle_dataset.py")
    else:
        success = setup_complete_workflow()
        if not success:
            manual_setup_instructions()
