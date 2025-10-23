# 🚗 Car Brand Classification — End-to-End (PyTorch + timm)

A complete deep learning pipeline for **Car Brand Classification**, built with **PyTorch** and **timm**’s EfficientNetV2-S architecture.  
The project covers **EDA → Preprocessing → Data Balancing → Model Training → Evaluation → Inference**, aligned with university ML coursework requirements.

---

## 🧠 Overview

This project classifies car images by brand (e.g., Audi, BMW, Ferrari) using transfer learning.  
It uses **EfficientNetV2-S pretrained on ImageNet**, fine-tuned on a Kaggle dataset of car brand images.

**Key highlights:**
- Automated data discovery and stratified splitting  
- Visual EDA (class counts, balance plots, confusion matrix)  
- Weighted sampling to handle class imbalance  
- Transfer learning with EfficientNetV2-S  
- Mixed precision training (AMP)  
- Macro F1–based evaluation and model checkpointing  
- Easy inference helper for new images

---

## 📂 Dataset

- **Source:** [Kaggle — Cars Image Dataset by Kshitij192](https://www.kaggle.com/datasets/kshitij192/cars-image-dataset)
- The original dataset contained `train/` and `test/` folders.  
  Both were **merged** into one dataset and re-split (70/15/15) for custom stratified training, validation, and testing.

| Property | Description |
|-----------|--------------|
| Classes | Multiple car brands (Audi, BMW, Ferrari, etc.) |
| Images | RGB `.jpg` / `.png` |
| Image Size | Standardized to 384×384 |
| Total | ~8,000 images |
| Split | 70% train, 15% validation, 15% test |

---

## ⚙️ Setup

### **1️⃣ Clone & Install**
```bash
git clone https://github.com/<your-username>/car-brand-classification-pytorch.git
cd car-brand-classification-pytorch
pip install -r requirements.txt
