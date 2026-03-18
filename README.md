# Comparative Analysis of Deep Learning Models for Skin Cancer Classification

## 📌 Overview
This project performs a comparative analysis of five deep learning models for multi-class skin cancer classification using the HAM10000 dataset.

Models evaluated:
- Custom CNN
- ResNet50
- InceptionV3
- Xception
- InceptionResNetV2

The objective was to compare performance using accuracy, precision, recall, and F1-score.

---

## 📊 Dataset
HAM10000 Dataset  
10,015 dermatoscopic images  
7 skin lesion classes:
- nv
- mel
- bkl
- bcc
- akiec
- vasc
- df

After deduplication: 7,470 images  
Split:
- 64% Training
- 16% Validation
- 20% Testing

---

## 🧠 Models Implemented

| Model | Accuracy |
|-------|----------|
| Custom CNN | 74% |
| ResNet50 | 75% |
| InceptionV3 | 84% |
| Xception | **85.63%** |
| InceptionResNetV2 | 83.62% |

Best Model: **Xception (85.63%)**

---

## ⚙️ Technologies Used
- Python
- PyTorch
- TensorFlow / Keras
- Scikit-learn
- Matplotlib
- KaggleHub

---

## 🚀 How to Run

1. Install dependencies: pip install -r requirements.txt
2. Download dataset: kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
3. Run the notebooks: CNN.ipynb, ResNet50_v2.ipynb, InceptionV3.ipynb, Xception.ipynb, InceptionResnetV2

---

## 📈 Results & Insights
- Xception achieved highest accuracy (85.63%)
- Transfer learning significantly outperformed custom CNN
- Class imbalance affected melanoma recall
- Advanced architectures performed better on minority classes

---
## 🌐 Streamlit Web Application

An interactive **Streamlit application** is included to demonstrate real-time skin lesion classification using the best-performing model (**Xception**).

### ✨ Features
- Upload a dermoscopic image and get instant predictions
- Displays:
  - Predicted class
  - Class description
  - Confidence score
  - Top-3 predictions with probabilities
- Lightweight and easy to run locally

---

### 🖼️ Application Preview

![Streamlit UI](sample_test_images/Streamlit_web_ui.png)

---

