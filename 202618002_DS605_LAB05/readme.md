# Lab 05 — Linear and Logistic Regression: Implementation from Scratch

**Course:** DS605 — Fundamentals of Machine Learning  
**Student:** Deeksha  
**Student ID:** 202618002

## Overview

This lab explores the mathematical foundations and practical implementation of Linear Regression and Logistic Regression using the Garment Worker Productivity dataset.

The main objective is to understand how regression algorithms work internally by implementing their optimisation procedures from scratch using NumPy, then comparing their predictions, evaluation metrics, and computational performance against Scikit-learn implementations.

The lab also investigates feature engineering, regularisation, and alternative optimisation methods to improve model performance.

## Dataset

**File:** `garments_worker_productivity.csv`

The dataset contains 1,197 records of garment factory team productivity.

Important features include:

- `quarter`, `department`, `day`, and `team`
- `targeted_productivity`
- `smv` (Standard Minute Value)
- `wip` (Work in Progress)
- `over_time` and `incentive`
- `idle_time` and `idle_men`
- `no_of_style_change`
- `no_of_workers`
- `actual_productivity`

## Objectives

1. Understand the mathematical foundations of Linear and Logistic Regression.
2. Build regression and classification models using Scikit-learn.
3. Implement Ordinary Least Squares (OLS) and Logistic Regression from scratch using NumPy.
4. Understand gradient descent and its convergence behaviour.
5. Explore feature engineering and Ridge regularisation.
6. Implement Logistic Regression using IRLS/Newton's method.
7. Compare model performance and computational efficiency.

## Methodology

### 1. Data Exploration and Preprocessing

- Loaded and inspected the dataset using Pandas.
- Examined data types, statistical summaries, and missing values.
- Standardised inconsistent department labels by removing leading and trailing whitespace.
- Removed the date column for modelling.
- Clipped `actual_productivity` values at 1.0 for the modelling target.
- Created a binary classification target named `MeetsTarget`, indicating whether actual productivity met or exceeded targeted productivity.
- Converted team identifiers into categorical values.
- Split the data into training and testing sets using an 80:20 stratified split with `random_state=42`.

### 2. Feature Preprocessing

- Separated numerical and categorical features.
- Imputed missing numerical values using the median.
- Imputed missing categorical values using the most frequent category.
- Applied StandardScaler to numerical features.
- Applied One-Hot Encoding to categorical features.
- Ensured preprocessing parameters were learned from training data and applied consistently to test data.

### 3. Part A — Scikit-learn Implementation

Built two baseline models:

**Linear Regression**
- Predicts continuous actual productivity.
- Evaluated using MAE, RMSE, and R².

**Logistic Regression**
- Predicts whether a team meets its productivity target.
- Evaluated using accuracy, precision, recall, and F1-score.

Training and prediction times were also measured.

### 4. Part B — Implementation from Scratch

**Ordinary Least Squares (OLS)**

Implemented Linear Regression using the normal equation:

\[
\beta=(X^TX)^{-1}X^Ty
\]

The implementation uses `numpy.linalg.solve()` rather than explicitly calculating the matrix inverse.

**Logistic Regression using Gradient Descent**

Implemented:
- Sigmoid activation function
- Probability prediction
- Binary cross-entropy loss
- Gradient calculation
- Iterative coefficient updates
- Convergence monitoring using the gradient magnitude

The scratch implementations were compared against Scikit-learn to verify numerical and predictive consistency.

### 5. Part C — Feature Engineering and Optimisation

Further improvements were explored through:

- Log transformations of selected features.
- Per-worker ratio features.
- Removal of near-duplicate or highly redundant features.
- Correlation analysis using Pearson and Spearman correlation.
- Ridge Regression with L2 regularisation.
- Cross-validation for selecting the Ridge regularisation parameter.
- Logistic Regression using IRLS/Newton's method.
- Five-fold cross-validation for selecting the logistic L2 regularisation parameter.

These experiments examined whether better feature representations and optimisation methods could improve predictive performance.

## Results

### Regression Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Scikit-learn Linear Regression | 0.1079 | 0.1423 | 0.2924 |
| Scratch OLS — Part B | 0.1079 | 0.1423 | 0.2924 |
| Scratch OLS — Part C | 0.1050 | 0.1410 | 0.3051 |
| Scratch Ridge — Part C | 0.1047 | 0.1400 | 0.3149 |

Feature engineering and Ridge regularisation produced a modest improvement in regression performance. The final Ridge model achieved an R² of approximately 0.315.

### Classification Performance

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Scikit-learn Logistic Regression | 0.6833 | 0.7619 | 0.8229 | 0.7912 |
| Scratch Logistic Regression — Part B | 0.6875 | 0.7688 | 0.8171 | 0.7922 |
| Scratch Logistic Regression — Part C | 0.6958 | 0.8000 | 0.7771 | 0.7884 |
| IRLS + L2 — CV λ | 0.6875 | 0.7778 | 0.8000 | 0.7887 |
| IRLS + L2 — λ = 1 | 0.7000 | 0.7977 | 0.7886 | 0.7931 |

Classification performance remained broadly similar across the implementations, with F1-scores around 0.79.

### Computational Performance

The notebook also compares training and prediction time.

Key observations:

- Scratch OLS was substantially faster than the Scikit-learn implementation in the recorded small-dataset benchmark.
- Basic scratch Logistic Regression using gradient descent required approximately 11,917 iterations and was slower than Scikit-learn.
- IRLS with L2 regularisation converged in approximately 7 iterations, reducing training time considerably.
- Runtime comparisons are specific to the recorded environment and dataset size; they should not be treated as universal performance guarantees.

## Key Learnings

- Understanding the normal equation and its relationship to OLS.
- Deriving and implementing gradient descent for Logistic Regression.
- Understanding sigmoid probabilities and binary cross-entropy.
- Learning how feature scaling and encoding affect model training.
- Understanding L1/L2 regularisation concepts and Ridge Regression.
- Comparing gradient-based optimisation with Newton/IRLS methods.
- Evaluating regression and classification models using appropriate metrics.
- Understanding the trade-off between predictive performance and computational efficiency.

## Repository Structure

```text
202618002_DS605_LAB05/
│
├── 202618002_DS605_lab05.ipynb
├── garments_worker_productivity.csv
└── readme.md
```

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## How to Run

1. Clone or download the repository.
2. Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

3. Open the notebook:

```bash
jupyter notebook 202618002_DS605_lab05.ipynb
```

4. Ensure `garments_worker_productivity.csv` is in the same directory as the notebook.
5. Execute the notebook cells sequentially.

---

**Author:** Deeksha  
**Student ID:** 202618002  
**Course:** DS605 — Fundamentals of Machine Learning
