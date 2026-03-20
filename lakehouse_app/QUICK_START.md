# Quick Start Guide - CV Training Pipeline App

Get started with the Computer Vision Training Pipeline in 5 minutes!

## 🎯 What You'll Learn

In this quick start, you'll:
1. Create your first training configuration
2. Explore a dataset
3. Launch a training job
4. View training metrics
5. Register and deploy a model

**Time Required:** ~15 minutes (plus training time)

## 🚦 Before You Start

Make sure you have:
- [ ] Access to the CV Training Pipeline app
- [ ] A dataset uploaded to Unity Catalog Volumes
- [ ] Basic knowledge of your CV task (detection, classification, etc.)

## 📝 Step-by-Step Tutorial

### Step 1: Create Your First Configuration (3 minutes)

1. **Open the app** and navigate to **⚙️ Config Setup**

2. **Select your task:**
   - For this example, let's use **Object Detection**
   - Click the dropdown and select "🔍 Object Detection"

3. **Choose a model:**
   - Select "DETR ResNet-50" for a good balance of speed and accuracy
   - Click the info icon to see model details

4. **Configure your data:**
   - **Training Images Path:** `/Volumes/main/cv_models/cv_data/data/train/`
   - **Training Annotations:** `/Volumes/main/cv_models/cv_data/data/train_annotations.json`
   - **Validation Images Path:** `/Volumes/main/cv_models/cv_data/data/val/`
   - **Validation Annotations:** `/Volumes/main/cv_models/cv_data/data/val_annotations.json`
   - **Number of Classes:** 80 (for COCO dataset, adjust for your dataset)

5. **Set training parameters:**
   - **Epochs:** 50 (start small for testing, increase later)
   - **Batch Size:** 16 (adjust based on GPU memory)
   - **Learning Rate:** 1e-4 (the default is usually good)

6. **Configure MLflow:**
   - **Experiment Name:** `/Users/your.email@company.com/cv_detection_test`
   - ⚠️ Make sure to use YOUR email address!

7. **Set output paths:**
   - **Results Directory:** `/Volumes/main/cv_models/cv_data/results/detection`
   - Leave other options as default

8. **Save your configuration:**
   - Click "✅ Save & Set as Active"
   - You'll see a success message with confetti! 🎉

### Step 2: Explore Your Data (2 minutes)

1. **Navigate to** **📊 Data EDA**

2. **View dataset statistics:**
   - Click "🔍 Analyze Dataset"
   - Review the number of images and annotations
   - Check the train/val split visualization

3. **Inspect class distribution:**
   - Go to the "🔍 Class Distribution" tab
   - Click "📊 Analyze Class Distribution"
   - Look for class imbalance (ratio > 10x might need attention)

4. **Preview samples:**
   - Go to "🖼️ Sample Viewer" tab
   - Click "🎲 Load Random Samples"
   - Verify your images and annotations look correct

5. **Run validation:**
   - Go to "✅ Data Validation" tab
   - Click "✅ Run Validation Checks"
   - Address any warnings or errors

### Step 3: Launch Training (5 minutes)

1. **Navigate to** **🚀 Training**

2. **Review your configuration:**
   - Check the summary at the top
   - Verify task, model, epochs, and batch size
   - Click "View Full Configuration" to see YAML

3. **Configure the job:**
   - **Job Name:** `my_first_cv_training`
   - **Source Path:** `/Workspace/Repos/<username>/databricks-cv-accelerator/src`
   - ⚠️ Replace `<username>` with your actual username!

4. **Select compute:**
   - Uncheck "Use Serverless Compute" (for GPU training)
   - **Node Type:** g5.4xlarge (1 GPU, good for getting started)
   - **Workers:** 0 (single-node training)

5. **Launch:**
   - Click "🚀 Launch Training Job"
   - Wait for confirmation (Job ID and Run ID)
   - Success! 🎉

6. **Monitor training:**
   - Switch to "📊 Monitor Training" tab
   - Enable "Auto-refresh" to see live updates
   - Watch the training metrics update in real-time

### Step 4: Evaluate Your Model (3 minutes)

**⏰ Wait for training to complete first (check Monitor Training tab)**

1. **Navigate to** **📈 Evaluation**

2. **Load your runs:**
   - Enter your experiment name: `/Users/your.email@company.com/cv_detection_test`
   - Click "🔍 Load Runs"
   - Select your completed run

3. **View metrics:**
   - Review the metrics summary (mAP, loss, etc.)
   - Expand "📈 Metrics" to see detailed numbers
   - Check "⚙️ Parameters" to see training settings

4. **Visualize training:**
   - Select metrics to visualize (e.g., val_map, val_loss)
   - View the training curves
   - Look for signs of overfitting or underfitting

### Step 5: Register Your Model (2 minutes)

1. **Navigate to** **📦 Model Registration**

2. **Select a training run:**
   - The experiment name is auto-filled from your config
   - Select the completed training run from the dropdown
   - The model URI and checkpoint directory are discovered automatically

3. **Fill in registration details:**
   - **Catalog:** main
   - **Schema:** cv_models
   - **Model Name:** `my_first_detection_model`
   - **Description:** "My first object detection model trained on COCO"
   - **Tags:** `detection,detr,coco`

4. **Register:**
   - Click "Register Model"
   - Wait for registration to complete
   - Success! Your model is now in Unity Catalog

### Step 6: Deploy Your Model (Optional, 3 minutes)

1. **Navigate to** **🌐 Deployment**

2. **Configure endpoint:**
   - Your model should be auto-selected
   - **Endpoint Name:** `my_detection_endpoint`
   - **Workload Size:** Small (for testing)
   - **Scale-to-Zero:** ✓ Enabled (to save costs)

3. **Deploy:**
   - Click "🚀 Deploy to Endpoint"
   - Wait for deployment (takes 5-10 minutes)
   - Check "📊 Active Endpoints" tab for status

### Step 7: Test Your Model (2 minutes)

1. **Navigate to** **🎮 Inference**

2. **Upload a test image:**
   - Click "Upload an image"
   - Select a test image from your computer

3. **Run inference:**
   - Adjust confidence threshold if needed (default: 0.5)
   - Click "🚀 Run Inference"
   - View the predictions!

4. **Download results:**
   - Click "📥 Download Annotated Image"
   - Save the results JSON

## 🎉 Congratulations!

You've successfully:
- ✅ Created a training configuration
- ✅ Explored your dataset
- ✅ Launched a training job
- ✅ Evaluated your model
- ✅ Registered your model
- ✅ Deployed to an endpoint (optional)
- ✅ Ran inference on test images

## 🚀 Next Steps

Now that you're familiar with the basics:

### Improve Your Model

1. **Tune hyperparameters:**
   - Experiment with learning rates (1e-5 to 1e-3)
   - Try different batch sizes
   - Adjust epochs based on convergence

2. **Try different models:**
   - YOLOS for faster inference
   - ResNet-101 for higher accuracy
   - Compare performance in Evaluation page

3. **Augment your data:**
   - Enable data augmentation in config
   - Adjust augmentation parameters
   - Add more training data

### Scale Up

1. **Distributed training:**
   - Use multi-GPU (g5.12xlarge with 4 GPUs)
   - Enable distributed training in config
   - Increase batch size accordingly

2. **Batch inference:**
   - Use the "⚙️ Batch Inference" tab
   - Process large datasets efficiently
   - Save results to Delta tables

3. **Production deployment:**
   - Promote model to Production stage
   - Scale endpoint to Medium/Large
   - Set up monitoring and alerts

## 💡 Pro Tips

1. **Start small:** Test with a small subset of data first
2. **Monitor costs:** Enable scale-to-zero for endpoints
3. **Version everything:** Keep track of configs and models
4. **Use History page:** Track all your experiments
5. **Check validation:** Always validate data before training

## 📚 Learn More

- **README.md** - Complete feature documentation
- **DEPLOYMENT_GUIDE.md** - Deployment and configuration
- **Docs folder** - Framework documentation

## 🆘 Common Issues

### "No active configuration found"
➡️ Go to Config Setup and save a configuration first

### "Job submission failed"
➡️ Check your source path is correct
➡️ Verify you have GPU quota available

### "Cannot access data path"
➡️ Verify Unity Catalog permissions
➡️ Check paths are correct (no `<placeholders>`)

### "Experiment not found"
➡️ Use absolute workspace paths: `/Users/email@company.com/experiment`
➡️ Create the experiment manually if needed

## 🎯 Quick Reference

### Recommended Settings by Model Size

| Model Size | Batch Size | Learning Rate | Epochs | GPU          |
|-----------|-----------|---------------|--------|--------------|
| Small     | 32        | 2e-4          | 100    | g5.4xlarge   |
| Medium    | 16        | 1e-4          | 100    | g5.4xlarge   |
| Large     | 8         | 5e-5          | 150    | g5.12xlarge  |

### Typical Training Times

| Dataset Size | Model    | GPUs | Time     |
|-------------|----------|------|----------|
| Small (5K)  | DETR-50  | 1    | 2-3 hrs  |
| Medium (50K)| DETR-50  | 1    | 8-12 hrs |
| Large (100K)| DETR-50  | 4    | 6-8 hrs  |

---

**Happy Training!** 🚀

Need help? Check the troubleshooting section in README.md or contact your workspace admin.

