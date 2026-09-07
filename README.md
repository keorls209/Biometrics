# Biometric Face Recognition and Verification System

A comprehensive face recognition and verification benchmark evaluating **PCA (Eigenfaces)**, **Local Binary Patterns (LBP)**, **Deep Convolutional Neural Networks (CNN)**, and **Feature-Level Fusion (PCA + LBP)** using Cosine Similarity, Euclidean Distance, and Support Vector Machines (SVM).

---

## 📌 Project Overview

This project implements and evaluates end-to-end biometric facial recognition and verification pipelines on the **Georgia Tech Face Database (`gt_db`)**:
1. **Preprocessing & Face Alignment**: Landmark detection and face normalization.
2. **Feature Extraction**:
   - **PCA (Eigenfaces)**: Dimensionality reduction capturing global facial variations.
   - **LBP (Local Binary Patterns)**: Texture-based micro-pattern representation.
   - **Deep CNN**: Deep representation features.
   - **Feature-Level Fusion**: Combining PCA and LBP representations with grid-searched optimal SVM.
3. **Matching & Classification**:
   - **Method A**: Cosine Similarity
   - **Method B**: Euclidean Distance
   - **Method C**: Support Vector Machine (SVM)
4. **Biometric Performance Evaluation**:
   - Rank-1 Recognition Accuracy
   - Equal Error Rate (EER)
   - Decidability Index ($d'$)
   - False Match Rate (FMR) & True Match Rate (TMR) at thresholds (e.g., FMR = 1%, FMR = 0.01%)
   - Genuine vs. Impostor score distributions
   - ROC (Receiver Operating Characteristic) curves

---

## 📂 Project Structure

```
.
├── TEAM 4/                  # Project report and documentation (PDF / Word)
├── data/                    # Extracted feature vectors (.npy) for PCA, LBP, and CNN
├── gt_db/                   # Georgia Tech Face Database images
├── cnn_predicted_vs_actual.png
├── cnn_top3_matches.png
├── gen_imp_cnn.png
├── gen_imp_lbp.png
├── gen_imp_pca.png
├── lbp_visualization.png
├── roc_cnn.png
├── roc_lbp.png
├── roc_pca.png
├── feature extraction.ipynb # Preprocessing and feature extraction pipeline
├── main.py                  # Evaluation & benchmark execution
├── matcher.py               # Matching algorithms (Cosine, Euclidean, SVM, Fusion)
├── matrix.py                # Biometric evaluation metrics (EER, D-prime, ROC, etc.)
└── .gitignore               # Ignored files (pycache, large weights, etc.)
```

---

## 🚀 Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install numpy scipy scikit-learn matplotlib opencv-python dlib
```

> **Note**: For landmark detection in `feature extraction.ipynb`, `shape_predictor_68_face_landmarks.dat` is downloaded automatically if not already present.

### Running the Evaluation

To evaluate all feature representations and generate the performance curves:

```bash
python main.py
```

---

## 📊 Results Summary

- **Feature-Level Fusion (PCA + LBP)** combined with hyperparameter-tuned SVM delivers robust verification performance.
- Visualizations for ROC curves, score distributions, and prediction samples are saved in the project root.

---

## 👥 Authors
- **Team 4** - Biometrics Course
