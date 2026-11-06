
# Lab 06 — Image-Based Crack Detection and Email Spam Classification

**Course:** DS605 — Fundamentals of Machine Learning  
**Author:** Deeksha  
**Student ID:** 202618002

## Overview

This lab explores two supervised machine learning tasks:

1. **Crack Detection:** Classifying images as containing cracks or not containing cracks using image processing, handcrafted features, and machine learning classifiers.
2. **Email Spam Classification:** Classifying emails as spam or non-spam using word-count features and machine learning models.

---

## Part A — Crack Detection

### Objective

Build a binary classification system to distinguish between cracked and non-cracked images using traditional computer vision techniques and supervised learning.

### Dataset

- Total images: 400
- Crack images: 200
- Non-crack images: 200
- Image dimensions: 448 × 448 × 3
- Classes are balanced (50% each).
- No unreadable images were detected.

### Image Processing

The following preprocessing and image-processing techniques were explored:

- Image resizing to 256 × 256.
- Conversion from RGB to grayscale.
- Gaussian smoothing to reduce noise.
- Image sharpening and intensity inversion.
- Histogram equalization and CLAHE for contrast enhancement.
- Sobel operators to calculate horizontal and vertical gradients.
- Gradient magnitude to capture edge strength.
- Canny edge detection with adaptive thresholds.
- Probabilistic Hough Transform to detect line segments.

### Feature Extraction

A total of 30 handcrafted features were extracted from each image.

| Feature group | Number of features |
|---|---:|
| Intensity and statistical features | 16 |
| Canny edge features | 2 |
| Gradient features | 6 |
| Hough line features | 6 |
| **Total** | **30** |

Features include brightness, contrast, intensity percentiles, entropy, dark-pixel ratio, edge density, Sobel gradient statistics, and Hough line characteristics.

Feature extraction took approximately 15.13 seconds for all 400 images, or 37.83 ms per image.

### Machine Learning Models

The dataset was divided using an 80:20 stratified train-test split with `random_state=42`.

The following classifiers were evaluated:

- Logistic Regression
- Support Vector Machine (RBF kernel)
- K-Nearest Neighbors
- Random Forest

Five-fold stratified cross-validation and GridSearchCV were used for model selection and hyperparameter tuning.

### Results

| Model | Test Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| SVM | 93.75% | 0.9268 | 0.9500 | 0.9383 |
| Random Forest | 93.75% | 0.9487 | 0.9250 | 0.9367 |
| Logistic Regression | 92.50% | 0.9250 | 0.9250 | 0.9250 |
| KNN | 92.50% | 0.9250 | 0.9250 | 0.9250 |

The SVM achieved the highest F1-score, while SVM and Random Forest achieved the highest test accuracy.

### Feature Ablation and Group Validation

Feature ablation experiments compared intensity, Canny, gradient, and Hough feature groups to understand their contribution to classification.

A further validation experiment grouped images using filename-derived capture groups. This was important because images from the same capture session may share visual characteristics.

The recorded group-validation F1 scores were lower than the corresponding random-fold scores:

| Model | Random-fold F1 | Group-fold F1 |
|---|---:|---:|
| Logistic Regression | 0.9543 | 0.8878 |
| SVM | 0.9478 | 0.8934 |
| Random Forest | 0.9555 | 0.9233 |

This indicates that random validation may give an optimistic estimate when related images appear in both training and validation folds.

---

## Part B — Email Spam Classification

### Objective

Develop a supervised classification system to identify spam emails using preprocessed word-frequency features.

### Dataset

- Total records: 5,172 emails.
- Input representation: 3,000 word-count features.
- Target column: `Prediction`.
- Class 0: Non-Spam.
- Class 1: Spam.
- Feature matrix sparsity: approximately 94.37%.

The dataset is already represented in Bag-of-Words form, where each email is represented by the occurrence counts of vocabulary terms.

### Data Preparation

- Removed `Email No.` from the input features.
- Separated the target variable `Prediction`.
- Used an 80:20 stratified train-test split.
- Applied `random_state=42`.

### Models Evaluated

Three classifiers were compared:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine

Evaluation metrics included accuracy, precision, recall, F1-score, training time, and prediction time.

### Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 98.26% | 0.9578 | 0.9833 | 0.9704 |
| Linear SVM | 97.97% | 0.9604 | 0.9700 | 0.9652 |
| Multinomial Naive Bayes | 94.20% | 0.8681 | 0.9433 | 0.9042 |

Logistic Regression achieved the highest test accuracy and F1-score among the three models.

### Feature Selection

Chi-square feature selection (`SelectKBest` with `chi2`) was used to investigate the effect of reducing the number of word features.

The number of selected features was varied across 300, 500, 1,000, 1,500, and 2,000.

The highest recorded mean cross-validation F1-score was approximately 0.9503 using 2,000 features.

This suggests that aggressive feature reduction can remove useful information, while retaining more informative word features can improve classification performance.

---

## Key Learnings

- Image preprocessing can improve the visibility of structural patterns before feature extraction.
- Sobel, Canny, and Hough Transform capture different aspects of image structure.
- Handcrafted image features can support effective classification without deep learning.
- Cross-validation and hyperparameter tuning help compare models systematically.
- Group-based validation can expose optimistic performance estimates caused by related samples.
- Bag-of-Words features provide a useful representation for traditional spam classification.
- Feature selection involves balancing computational efficiency and predictive performance.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- SciPy
- scikit-image
- scikit-learn

## Limitations

- The crack detection experiment uses a relatively small dataset of 400 images.
- Performance on images from completely different environments, cameras, or surfaces has not been established.
- The filename-based capture grouping is an approximation of image acquisition groups.
- The email dataset is already represented as word counts; raw email text processing is not demonstrated in this experiment.
- The reported metrics are based on the particular train-test splits and validation procedures used in the notebook.
