# Lab 07 — Face Recognition using PCA and Machine Learning

**Course:** DS605 — Fundamentals of Machine Learning  
**Author:** Deeksha  
**Student ID:** 202618002

## Overview

This lab implements a face recognition pipeline using the ORL face dataset, Principal Component Analysis (PCA), and traditional machine learning classifiers.

The experiment investigates how high-dimensional face images can be represented using a smaller number of principal components while retaining important facial information for recognition.

The notebook covers image representation, PCA through Singular Value Decomposition (SVD), eigenfaces, reconstruction, nearest-neighbor recognition, and classifier comparisons.

---

## Part A — Image Representation and Dataset Preparation

### Dataset

The experiment uses the ORL face dataset.

- Total images: 400
- Number of subjects: 40
- Images per subject: 10
- Image dimensions: 112 × 92 pixels
- Grayscale images
- Flattened feature dimensions: 10,304 pixels per image

Each image is converted into a one-dimensional feature vector.

The resulting dataset has:

- Feature matrix: `(400, 10304)`
- Target vector: `(400,)`

Each target represents the corresponding subject identity.

### Train-Test Split

A stratified split was performed separately for each subject:

- Training images: 8 per subject (320 total).
- Testing images: 2 per subject (80 total).
- Random seed: 42.

This ensures that every subject is represented in both training and testing sets.

---

## Part B — Principal Component Analysis and Eigenfaces

### Objective

Reduce the dimensionality of face images while preserving most of the information represented by the training data.

### PCA Implementation

The following steps were performed:

1. Calculate the mean face using the training images.
2. Subtract the mean face from each training image.
3. Apply Singular Value Decomposition (SVD).
4. Calculate eigenvalues and explained variance.
5. Determine the number of principal components required to retain 90% of the variance.
6. Visualize the principal components as eigenfaces.

The experiment independently cross-checks the SVD results using eigendecomposition of the smaller sample-space covariance matrix.

### PCA Results

The number of components required to retain at least 90% of the training variance was:

**99 principal components**, retaining approximately **90.0121%** of the variance.

| Representation | Dimensions |
|---|---:|
| Original image | 10,304 |
| PCA representation | 99 |

The dimensionality per face was reduced by approximately 99.04%.

### Reconstruction

Images were reconstructed from their PCA representations using different numbers of principal components.

Recorded reconstruction errors:

- Training reconstruction MSE: 154.13
- Testing reconstruction MSE: 323.21

The notebook also compares reconstruction quality for different component counts.

### Storage Analysis

The notebook compares the storage required for original images and the PCA representation.

| Component | Storage |
|---|---:|
| Original images (uint8) | 3,220 KB |
| PCA coefficients (float32) | 123.75 KB |
| PCA basis (float32) | 3,984.75 KB |
| Mean face (float32) | 40.25 KB |
| Total PCA representation | 4,148.75 KB |

Although PCA reduces the dimensionality of each image representation, the total storage in this experiment is higher because the PCA basis must also be stored.

---

## Part C — Nearest-Neighbor Face Recognition

### Objective

Evaluate whether PCA representations can be used to identify a subject by comparing a test face with known training faces.

### Distance Metrics

Nearest-neighbor recognition was implemented using:

- Euclidean distance on raw pixel features.
- Euclidean distance on PCA features.
- Weighted Euclidean distance on PCA features.

The weighted distance uses eigenvalue-based feature weights controlled by the parameter alpha.

The alpha values tested were:

`0.0, 0.25, 0.5, 0.75, 1.0`

The best validation accuracy was 97.50%, achieved by both alpha = 0 and alpha = 0.25. The predefined tie-breaking rule selected alpha = 0 as the performance-optimal setting. Alpha = 0.25 was retained as a separate mild-weighting comparison.

### Recognition Results

| Method | Features | Correct Predictions | Test Accuracy |
|---|---:|---:|---:|
| Raw Pixels + Euclidean | 10,304 | 79/80 | 98.75% |
| PCA + Euclidean | 99 | 79/80 | 98.75% |
| PCA + Weighted Euclidean (alpha = 0.25) | 99 | 79/80 | 98.75% |

PCA achieved the same test accuracy as raw-pixel nearest-neighbor recognition while reducing the feature dimensions substantially.

---

## Part D — Classifier Comparison

### Objective

Compare multiple machine learning classifiers using PCA-transformed face representations.

The notebook evaluates:

- K-Nearest Neighbors
- Linear Support Vector Machine
- RBF Support Vector Machine
- Logistic Regression

### Hyperparameter Experiments

Validation experiments investigated:

- Different K values for KNN.
- Different C values for linear SVM.
- Different C and gamma combinations for RBF SVM.
- Different regularization strengths for Logistic Regression.
- Different PCA component counts.

The final component counts were selected using validation performance, with ties resolved in favour of the smaller component count.

### Final Test Results

| Model | PCA Features | Correct Predictions | Test Accuracy |
|---|---:|---:|---:|
| KNN (K = 1) | 40 | 79/80 | 98.75% |
| Linear SVM | 100 | 79/80 | 98.75% |
| RBF SVM | 40 | 79/80 | 98.75% |
| Logistic Regression | 84 | 78/80 | 97.50% |

KNN, Linear SVM, and RBF SVM achieved the highest recorded test accuracy of 98.75%.

---

## Key Learnings

- A grayscale image can be represented as a high-dimensional numerical feature vector.
- Mean-centering is an important step before applying PCA.
- SVD provides an efficient way to obtain principal components.
- Eigenfaces represent directions of variation across the training face images.
- PCA can substantially reduce feature dimensionality while preserving much of the training variance.
- Reconstruction error helps evaluate information lost during dimensionality reduction.
- Nearest-neighbor methods can perform face recognition using distances between feature representations.
- SVM, KNN, and Logistic Regression provide different approaches to classification in PCA-transformed feature spaces.
- Hyperparameter selection should be performed using validation data rather than the final test set.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- OpenCV
- scikit-learn

## Limitations

- The experiment uses a relatively small dataset of 400 images and 40 subjects.
- The recognition task assumes fixed-size, grayscale face images.
- The notebook does not demonstrate a complete face-detection system that locates faces in arbitrary photographs or video streams.
- The test set contains images of subjects already represented in the training set. Therefore, the results measure recognition among known identities, not identification of previously unseen people.
- The reported 98.75% accuracy is based on 80 held-out images from this dataset and should not be interpreted as general-purpose face recognition accuracy.
- PCA preserves variance, but high variance does not necessarily correspond entirely to identity-specific information.
