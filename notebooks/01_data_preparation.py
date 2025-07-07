# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # 01. Data Preparation for DETR Training
# MAGIC 
# MAGIC This notebook prepares the COCO dataset for training DETR (DEtection TRansformer) on object detection. We'll explore the dataset structure, validate annotations, set up data loaders, and ensure everything is ready for training.
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC **COCO (Common Objects in Context)** is the standard benchmark dataset for object detection. It contains:
# MAGIC - **118K images** across train/val/test splits
# MAGIC - **80 object categories** (person, car, dog, etc.)
# MAGIC - **2.5M instances** with bounding boxes and segmentation masks
# MAGIC - **Rich annotations** including keypoints, captions, and panoptic segmentation
# MAGIC 
# MAGIC ### COCO Format for DETR:
# MAGIC DETR expects COCO format annotations with:
# MAGIC - **Bounding boxes**: [x, y, width, height] in pixel coordinates
# MAGIC - **Class labels**: Integer IDs corresponding to COCO categories
# MAGIC - **Image metadata**: File paths, dimensions, and IDs
# MAGIC 
# MAGIC ## What This Notebook Does
# MAGIC 
# MAGIC 1. **Dataset Exploration**: Understand COCO structure and statistics
# MAGIC 2. **Annotation Validation**: Verify COCO format compliance
# MAGIC 3. **Data Loading**: Set up efficient data loaders with preprocessing
# MAGIC 4. **Visualization**: Sample and visualize training data
# MAGIC 5. **Performance Optimization**: Configure for multi-GPU training
# MAGIC 
# MAGIC ---

# COMMAND ----------

# (Databricks only) Install requirements and restart Python if running interactively
# %pip install -r "../requirements.txt"
# dbutils.library.restartPython()
# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Import Dependencies and Load Configuration

# COMMAND ----------

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
from pathlib import Path

# Add the src directory to Python path
sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref/src')

from config import load_config
from tasks.detection.data import DetectionDataModule
from tasks.detection.adapters import get_input_adapter
from utils.logging import setup_logger

# Load configuration from previous notebook
# (In a real scenario, you'd load this from the previous notebook's output)
CATALOG = "your_catalog"
SCHEMA = "your_schema"  
VOLUME = "your_volume"
PROJECT_PATH = "cv_detr_training"

BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs/detection_detr_config.yaml"

# Load configuration
if os.path.exists(CONFIG_PATH):
    config = load_config(CONFIG_PATH)
else:
    print("⚠️  Config file not found. Using default detection config.")
    from config import get_default_config
    config = get_default_config("detection")

print("✅ Configuration loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. COCO Dataset Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and Analyze COCO Annotations

# COMMAND ----------

def analyze_coco_dataset(annotation_file: str, split_name: str):
    """Analyze COCO dataset statistics."""
    
    if not os.path.exists(annotation_file):
        print(f"❌ Annotation file not found: {annotation_file}")
        return None
    
    print(f"\n📊 Analyzing {split_name} dataset...")
    
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Basic statistics
    num_images = len(coco_data['images'])
    num_annotations = len(coco_data['annotations'])
    num_categories = len(coco_data['categories'])
    
    print(f"📈 Dataset Statistics:")
    print(f"   Images: {num_images:,}")
    print(f"   Annotations: {num_annotations:,}")
    print(f"   Categories: {num_categories}")
    
    # Category analysis
    category_counts = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
    
    # Get category names
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    print(f"\n🏷️  Top 10 Categories by Instance Count:")
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    for cat_id, count in sorted_categories[:10]:
        cat_name = cat_id_to_name.get(cat_id, f"Unknown-{cat_id}")
        print(f"   {cat_name}: {count:,} instances")
    
    # Image size analysis
    widths = [img['width'] for img in coco_data['images']]
    heights = [img['height'] for img in coco_data['images']]
    
    print(f"\n📏 Image Size Statistics:")
    print(f"   Width - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.1f}")
    print(f"   Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.1f}")
    print(f"   Aspect Ratio - Min: {min(w/h for w, h in zip(widths, heights)):.2f}, Max: {max(w/h for w, h in zip(widths, heights)):.2f}")
    
    # Instances per image
    instances_per_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        instances_per_image[img_id] = instances_per_image.get(img_id, 0) + 1
    
    instance_counts = list(instances_per_image.values())
    print(f"\n🎯 Instances per Image:")
    print(f"   Min: {min(instance_counts)}, Max: {max(instance_counts)}")
    print(f"   Mean: {np.mean(instance_counts):.1f}, Median: {np.median(instance_counts):.1f}")
    print(f"   Std: {np.std(instance_counts):.1f}")
    
    return coco_data

# Analyze train and validation sets
train_data = analyze_coco_dataset(config['data']['train_annotation_file'], "Train")
val_data = analyze_coco_dataset(config['data']['val_annotation_file'], "Validation")

# COMMAND ----------

# MAGIC %md
# MAGIC ### COCO Category Mapping

# COMMAND ----------

def display_coco_categories(coco_data):
    """Display all COCO categories with their IDs."""
    
    if not coco_data:
        return
    
    categories = coco_data['categories']
    print(f"\n📋 COCO Categories ({len(categories)} total):")
    
    # Group categories by supercategory
    supercategories = {}
    for cat in categories:
        supercat = cat['supercategory']
        if supercat not in supercategories:
            supercategories[supercat] = []
        supercategories[supercat].append(cat)
    
    for supercat, cats in supercategories.items():
        print(f"\n🏷️  {supercat}:")
        for cat in cats:
            print(f"   {cat['id']:2d}: {cat['name']}")

if train_data:
    display_coco_categories(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Validation and Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate COCO Format Compliance

# COMMAND ----------

def validate_coco_format(coco_data, split_name: str):
    """Validate that COCO data follows the expected format."""
    
    print(f"\n🔍 Validating {split_name} COCO format...")
    
    required_image_fields = ['id', 'file_name', 'width', 'height']
    required_annotation_fields = ['id', 'image_id', 'category_id', 'bbox']
    required_category_fields = ['id', 'name', 'supercategory']
    
    # Validate images
    image_errors = 0
    for img in coco_data['images']:
        for field in required_image_fields:
            if field not in img:
                print(f"❌ Image missing field: {field}")
                image_errors += 1
    
    # Validate annotations
    annotation_errors = 0
    for ann in coco_data['annotations']:
        for field in required_annotation_fields:
            if field not in ann:
                print(f"❌ Annotation missing field: {field}")
                annotation_errors += 1
        
        # Validate bbox format [x, y, width, height]
        if 'bbox' in ann:
            bbox = ann['bbox']
            if len(bbox) != 4:
                print(f"❌ Invalid bbox format: {bbox}")
                annotation_errors += 1
    
    # Validate categories
    category_errors = 0
    for cat in coco_data['categories']:
        for field in required_category_fields:
            if field not in cat:
                print(f"❌ Category missing field: {field}")
                category_errors += 1
    
    total_errors = image_errors + annotation_errors + category_errors
    
    if total_errors == 0:
        print(f"✅ {split_name} COCO format is valid!")
    else:
        print(f"❌ {split_name} has {total_errors} format errors")
    
    return total_errors == 0

# Validate both splits
train_valid = validate_coco_format(train_data, "Train") if train_data else False
val_valid = validate_coco_format(val_data, "Validation") if val_data else False

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Data Loading and Preprocessing

# COMMAND ----------

def test_data_loading():
    """Test data loading and preprocessing with a small sample."""
    
    print("\n🧪 Testing data loading and preprocessing...")
    
    try:
        # Create data module
        data_module = DetectionDataModule(config)
        
        # Test train dataloader
        train_loader = data_module.train_dataloader()
        print(f"✅ Train dataloader created successfully")
        print(f"   Batch size: {config['data']['batch_size']}")
        print(f"   Number of workers: {config['data']['num_workers']}")
        
        # Test validation dataloader
        val_loader = data_module.val_dataloader()
        print(f"✅ Validation dataloader created successfully")
        
        # Test a single batch
        sample_batch = next(iter(train_loader))
        print(f"✅ Sample batch loaded successfully")
        print(f"   Batch keys: {list(sample_batch.keys())}")
        
        # Check tensor shapes
        if 'pixel_values' in sample_batch:
            print(f"   Image tensor shape: {sample_batch['pixel_values'].shape}")
        if 'labels' in sample_batch:
            print(f"   Labels tensor shape: {sample_batch['labels'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

data_loading_success = test_data_loading()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Sample Images with Annotations

# COMMAND ----------

def visualize_sample_images(coco_data, image_dir, num_samples=3):
    """Visualize sample images with their bounding box annotations."""
    
    if not coco_data:
        print("❌ No COCO data available for visualization")
        return
    
    # Get sample images
    sample_images = coco_data['images'][:num_samples]
    
    # Create category mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Create image_id to annotations mapping
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    if num_samples == 1:
        axes = [axes]
    
    for i, img_info in enumerate(sample_images):
        img_id = img_info['id']
        img_path = os.path.join(image_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        ax = axes[i]
        
        # Display image
        ax.imshow(img)
        ax.set_title(f"Image {img_id}\n{img_info['file_name']}")
        ax.axis('off')
        
        # Draw bounding boxes
        if img_id in img_to_anns:
            for ann in img_to_anns[img_id]:
                bbox = ann['bbox']  # [x, y, width, height]
                cat_id = ann['category_id']
                cat_name = cat_id_to_name.get(cat_id, f"Unknown-{cat_id}")
                
                # Create rectangle patch
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(bbox[0], bbox[1]-5, cat_name, 
                       color='red', fontsize=8, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Visualized {len(sample_images)} sample images with annotations")

# Visualize sample images from train set
if train_data:
    visualize_sample_images(train_data, config['data']['train_data_path'], num_samples=3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Data Distribution

# COMMAND ----------

def analyze_data_distribution(coco_data, split_name: str):
    """Analyze and visualize data distribution."""
    
    if not coco_data:
        return
    
    # Category distribution
    category_counts = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
    
    # Get top categories
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    
    # Plot category distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of top categories
    categories = [cat_id_to_name[cat_id] for cat_id, count in top_categories]
    counts = [count for cat_id, count in top_categories]
    
    ax1.bar(range(len(categories)), counts, color='skyblue')
    ax1.set_title(f'{split_name} - Top 15 Categories')
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Number of Instances')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    
    # Histogram of instances per image
    instances_per_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        instances_per_image[img_id] = instances_per_image.get(img_id, 0) + 1
    
    instance_counts = list(instances_per_image.values())
    ax2.hist(instance_counts, bins=30, color='lightcoral', alpha=0.7)
    ax2.set_title(f'{split_name} - Instances per Image')
    ax2.set_xlabel('Number of Instances')
    ax2.set_ylabel('Number of Images')
    ax2.axvline(np.mean(instance_counts), color='red', linestyle='--', 
                label=f'Mean: {np.mean(instance_counts):.1f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Data distribution analysis completed for {split_name}")

# Analyze distributions
if train_data:
    analyze_data_distribution(train_data, "Train")
if val_data:
    analyze_data_distribution(val_data, "Validation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Performance Optimization

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure Data Loading for Multi-GPU Training

# COMMAND ----------

def optimize_data_loading():
    """Optimize data loading configuration for multi-GPU training."""
    
    print("\n⚡ Optimizing data loading for multi-GPU training...")
    
    # Get GPU count
    num_gpus = torch.cuda.device_count()
    print(f"   Available GPUs: {num_gpus}")
    
    # Calculate optimal batch size per GPU
    total_batch_size = config['data']['batch_size']
    batch_size_per_gpu = total_batch_size // num_gpus
    
    print(f"   Total batch size: {total_batch_size}")
    print(f"   Batch size per GPU: {batch_size_per_gpu}")
    
    # Optimize number of workers
    cpu_count = os.cpu_count()
    optimal_workers = min(cpu_count, 4 * num_gpus)  # 4 workers per GPU
    
    print(f"   CPU cores available: {cpu_count}")
    print(f"   Optimal workers: {optimal_workers}")
    
    # Update configuration
    config['data']['batch_size'] = total_batch_size
    config['data']['num_workers'] = optimal_workers
    
    # Test optimized configuration
    try:
        data_module = DetectionDataModule(config)
        
        # Test train dataloader
        train_loader = data_module.train_dataloader()
        print(f"✅ Optimized train dataloader created")
        print(f"   Batch size: {config['data']['batch_size']}")
        print(f"   Workers: {config['data']['num_workers']}")
        print(f"   Persistent workers: True")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return False

optimization_success = optimize_data_loading()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Memory Usage Analysis

# COMMAND ----------

def analyze_memory_usage():
    """Analyze memory usage for data loading."""
    
    print("\n💾 Analyzing memory usage...")
    
    try:
        # Create data module
        data_module = DetectionDataModule(config)
        
        # Test memory usage with a few batches
        train_loader = data_module.train_dataloader()
        
        # Get GPU memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / 1e9
            
            # Load a few batches
            for i, batch in enumerate(train_loader):
                if i >= 3:  # Test with 3 batches
                    break
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            memory_after = torch.cuda.memory_allocated() / 1e9
            memory_used = memory_after - memory_before
            
            print(f"   GPU memory before: {memory_before:.2f} GB")
            print(f"   GPU memory after: {memory_after:.2f} GB")
            print(f"   Memory used: {memory_used:.2f} GB")
            
            # Estimate memory per batch
            memory_per_batch = memory_used / 3
            print(f"   Estimated memory per batch: {memory_per_batch:.2f} GB")
            
            # Check if we have enough memory
            total_gpu_memory = sum([torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]) / 1e9
            print(f"   Total GPU memory: {total_gpu_memory:.2f} GB")
            
            if memory_per_batch * num_gpus < total_gpu_memory * 0.8:  # Leave 20% buffer
                print("✅ Memory configuration looks good!")
            else:
                print("⚠️  Memory usage might be too high. Consider reducing batch size.")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory analysis failed: {e}")
        return False

memory_analysis_success = analyze_memory_usage()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary and Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preparation Summary

# COMMAND ----------

print("=" * 60)
print("DATA PREPARATION SUMMARY")
print("=" * 60)

print(f"✅ COCO Dataset Analysis:")
print(f"   Train images: {len(train_data['images']) if train_data else 'N/A'}")
print(f"   Val images: {len(val_data['images']) if val_data else 'N/A'}")
print(f"   Categories: {len(train_data['categories']) if train_data else 'N/A'}")

print(f"\n✅ Data Validation:")
print(f"   Train format valid: {'Yes' if train_valid else 'No'}")
print(f"   Val format valid: {'Yes' if val_valid else 'No'}")

print(f"\n✅ Data Loading:")
print(f"   Loading test: {'Passed' if data_loading_success else 'Failed'}")
print(f"   Optimization: {'Passed' if optimization_success else 'Failed'}")
print(f"   Memory analysis: {'Passed' if memory_analysis_success else 'Failed'}")

print(f"\n🔧 Configuration:")
print(f"   Batch size: {config['data']['batch_size']}")
print(f"   Workers: {config['data']['num_workers']}")
print(f"   Image size: {config['data']['image_size']}")
print(f"   Augmentation: {'Enabled' if config['data']['augment'] else 'Disabled'}")

if all([train_valid, val_valid, data_loading_success, optimization_success]):
    print("\n🎉 DATA PREPARATION COMPLETE! Ready for training.")
    print("\nNext steps:")
    print("1. Run notebook 02_model_training.py to start training")
    print("2. Monitor training progress in MLflow UI")
    print("3. Check GPU utilization and memory usage")
else:
    print("\n⚠️  DATA PREPARATION INCOMPLETE! Please resolve issues before proceeding.")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Understanding DETR Data Requirements
# MAGIC 
# MAGIC ### COCO Format for DETR:
# MAGIC 
# MAGIC DETR expects COCO format annotations with specific requirements:
# MAGIC 
# MAGIC 1. **Bounding Boxes**: [x, y, width, height] in pixel coordinates
# MAGIC 2. **Class Labels**: Integer IDs (0-79 for COCO)
# MAGIC 3. **Image Metadata**: File paths, dimensions, and unique IDs
# MAGIC 4. **Consistent Format**: All annotations must follow COCO standard
# MAGIC 
# MAGIC ### Preprocessing Pipeline:
# MAGIC 
# MAGIC 1. **Image Resizing**: DETR resizes images to 800px shortest side
# MAGIC 2. **Normalization**: ImageNet mean/std normalization
# MAGIC 3. **Augmentation**: Random crops, flips, color jittering
# MAGIC 4. **Padding**: Images padded to largest size in batch
# MAGIC 5. **Target Formatting**: Annotations converted to DETR format
# MAGIC 
# MAGIC ### Key Considerations:
# MAGIC 
# MAGIC - **Memory Efficiency**: Batch size must fit in GPU memory
# MAGIC - **Data Loading**: Multiple workers for efficient loading
# MAGIC - **Validation**: Regular format checks prevent training errors
# MAGIC - **Visualization**: Helps debug data loading issues
# MAGIC 
# MAGIC The data preparation ensures that COCO annotations are properly formatted and efficiently loaded for DETR training! 