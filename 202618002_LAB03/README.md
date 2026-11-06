# DS605 Lab 03 — Hotel Booking Cancellation Prediction

## Overview

This lab focuses on preparing a real-world hotel booking dataset and building supervised machine learning models to predict whether a hotel reservation will be canceled.

The workflow covers data inspection, missing-value analysis, outlier detection, feature preprocessing, classification, and model evaluation. It also compares the effect of two feature-scaling techniques — StandardScaler and MinMaxScaler — on Logistic Regression and Decision Tree classifiers.

## Dataset

**Dataset:** `hotel_bookings.csv`

The dataset contains hotel reservation records with information such as:

- Hotel type
- Lead time
- Arrival date
- Length of stay
- Number of adults, children, and babies
- Previous cancellations and bookings
- Booking changes
- Average Daily Rate (ADR)
- Deposit type
- Customer type
- Reservation status

**Target variable:** `is_canceled`

- `0` — Booking not canceled
- `1` — Booking canceled

## Technologies and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Key Scikit-learn modules used:

- `train_test_split`
- `SimpleImputer`
- `KNNImputer`
- `StandardScaler`
- `MinMaxScaler`
- `OneHotEncoder`
- `ColumnTransformer`
- `Pipeline`
- `LogisticRegression`
- `DecisionTreeClassifier`
- Classification evaluation metrics

## Lab Workflow

### 1. Data Loading and Initial Exploration

- Load the hotel bookings dataset using Pandas.
- Inspect the first and last records.
- Examine dataset dimensions, column names, data types, and statistical summaries.
- Check the target variable and its class distribution.

### 2. Missing Values and Feature Selection

- Calculate missing-value counts and percentages.
- Identify columns with excessive missing values.
- Remove `company` because approximately 94.3% of its values are missing.
- Remove `reservation_status` and `reservation_status_date` to avoid target leakage, since these fields reveal information about the final reservation outcome.

### 3. Outlier Detection and Data Cleaning

- Identify potential outliers using the Interquartile Range (IQR) method.
- Calculate lower and upper outlier boundaries.
- Examine box plots for numerical features.
- Investigate bookings with zero adults, children, and babies.
- Remove records with no guests.
- Investigate extreme Average Daily Rate (ADR) values.
- Remove records with negative ADR and ADR values above 5000.

### 4. Train-Test Split and Preprocessing

Separate the dataset into:

- **Features (`X`):** all input variables except `is_canceled`.
- **Target (`y`):** the booking cancellation indicator.

Use an 80:20 train-test split with `random_state=42` and stratification to preserve the cancellation-class distribution.

Two preprocessing pipelines are constructed:

**Numerical features**

- KNN Imputation with 5 neighbors.
- StandardScaler in Pipeline A.
- MinMaxScaler in Pipeline B.

**Categorical features**

- Most-frequent-value imputation.
- OneHotEncoder with `handle_unknown="ignore"`.

A ColumnTransformer combines numerical and categorical preprocessing.

### 5. Classification Models

Train and compare four model configurations:

| Model | Numerical scaling |
|---|---|
| Logistic Regression A | StandardScaler |
| Logistic Regression B | MinMaxScaler |
| Decision Tree A | StandardScaler |
| Decision Tree B | MinMaxScaler |

The preprocessing and classification steps are combined using Scikit-learn Pipelines.

### 6. Model Evaluation and Comparison

Evaluate the models using:

- Training accuracy
- Testing accuracy
- Precision
- Recall
- F1-score
- Train-test accuracy gap

The train-test gap helps identify potential overfitting.

Confusion matrices are also generated for the best Logistic Regression and Decision Tree configurations.

## Key Observations

1. **Logistic Regression with StandardScaler performed best overall** among the tested configurations, based on the notebook's comparison of test F1-score and other evaluation metrics.

2. **Logistic Regression showed a smaller train-test gap**, with approximately 82.9% training accuracy and 82.5% testing accuracy.

3. **The Decision Tree showed signs of overfitting.** Without a maximum-depth constraint, it achieved approximately 99.6% training accuracy but around 86.3% testing accuracy.

4. **Feature scaling had little effect on Decision Tree performance.** Decision Trees use feature thresholds and ordering rather than distances between scaled values.

5. **The Logistic Regression confusion matrix** showed:
   - 13,596 correctly classified non-canceled bookings.
   - 6,171 correctly classified canceled bookings.
   - 2,669 actual cancellations incorrectly classified as non-canceled.

6. The results indicate that overall accuracy alone is not sufficient for evaluating cancellation prediction. Recall and F1-score are important because the model must also identify bookings that are actually canceled.

## Repository Structure

```text
202618002_LAB03/
│
├── 202618002_DS605_lab03 (1).ipynb
├── hotel_bookings.csv
├── Confusion matrix1.png
├── Confusion matrix 2.png
└── README.md
```

## How to Run

1. Clone or download the repository.
2. Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

3. Open the notebook:

```bash
jupyter notebook
```

4. Ensure `hotel_bookings.csv` is available in the working directory.
5. Update the dataset loading path in the notebook if necessary:

```python
df = pd.read_csv("hotel_bookings.csv")
```

6. Run the notebook cells sequentially.

## Learning Outcomes

- Performing exploratory data analysis on a real-world dataset.
- Identifying and handling missing values.
- Detecting and treating outliers using the IQR method.
- Understanding target leakage and its effect on model evaluation.
- Applying numerical and categorical preprocessing.
- Building preprocessing pipelines using ColumnTransformer and Pipeline.
- Understanding the differences between StandardScaler and MinMaxScaler.
- Implementing Logistic Regression and Decision Tree classifiers.
- Evaluating classification models using multiple performance metrics.
- Identifying overfitting through training and testing performance.

## Author

**Name:** Deeksha  
**Student ID:** 202618002  
**Course:** DS605 — Fundamentals of Machine Learning  
**Lab:** 03
