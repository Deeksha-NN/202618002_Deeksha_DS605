# Lab 08: Decision Trees and Support Vector Machines

**Course:** DS605 – Fundamentals of Machine Learning  
**Dataset:** UCI Banknote Authentication  
**Objective:** Implement Decision Trees and Linear SVMs from scratch, compare them with Scikit-learn, and explore nonlinear SVM kernels.

## 1. Overview

This lab explores two supervised machine learning algorithms: Decision Trees and Support Vector Machines (SVMs).

The objective is to understand their mathematical foundations, implement core algorithms using NumPy, visualize their decision boundaries, and compare manual implementations against Scikit-learn.

The experiments also investigate nonlinear classification using polynomial and RBF kernels.

## 2. Dataset

**UCI Banknote Authentication Dataset**

- Total observations: 1,372
- Input features: 4
- Target classes: 2
- Missing values: 0
- Duplicate records: 24

The four input features are:
1. Variance
2. Skewness
3. Curtosis
4. Entropy

Exact duplicate records were removed using `drop_duplicates()`, retaining the first occurrence of each record. The resulting dataset contained 1,348 unique observations.

An 80:20 stratified train-test split was used, resulting in 1,078 training samples and 270 testing samples.

## 3. Implementation

### Part A – Decision Tree from Scratch

Implemented a Decision Tree classifier using NumPy and Pandas.

- Gini impurity and entropy calculations.
- Feature and threshold selection using weighted impurity.
- Recursive tree construction.
- Stopping conditions and maximum-depth control.
- Prediction using learned split rules.
- Visualization of the manual tree and decision boundaries.

The manual implementation was compared with Scikit-learn's `DecisionTreeClassifier` using different criteria and maximum-depth settings.

### Part B – Linear SVM from Scratch

Implemented a binary soft-margin Linear SVM using NumPy.

- Standardization using training-set statistics.
- Binary label conversion from 0/1 to -1/+1.
- Hinge loss and regularization.
- Full-batch subgradient descent.
- Prediction using learned weights and bias.
- Visualization of the separating hyperplane, margins, and margin violations.

The manual implementation was compared with Scikit-learn's Linear SVM. Experiments investigated the effects of regularization parameter C, learning rate, and number of iterations.

### Part C – Kernel SVM and Nonlinear Boundaries

Generated a synthetic nonlinear dataset using `make_moons`.

Three Scikit-learn SVM models were trained:

- Linear kernel
- Polynomial kernel (degree 3)
- RBF kernel

Their decision boundaries and classification performance were compared to understand how kernel functions help classify nonlinear data.

Stratified 5-fold cross-validation was also performed on the training data.

### Part D – Final Four-Feature Comparison

All five models were evaluated using the original Banknote dataset with all four input features.

Models compared:

1. Manual Decision Tree
2. Scikit-learn Decision Tree
3. Manual Linear SVM
4. Scikit-learn Linear SVM
5. Scikit-learn RBF SVM

Evaluation metrics included:

- Accuracy
- Precision
- Recall
- F1-score
- Training time
- Prediction time

Additional experiments examined model complexity and used stratified 5-fold cross-validation on the training dataset.

## 4. Final Model Comparison

Held-out test-set results:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Manual Decision Tree | 94.44% | 90.84% | 97.54% | 0.9407 |
| Scikit-learn Decision Tree | 94.44% | 90.84% | 97.54% | 0.9407 |
| Manual Linear SVM | 90.74% | 95.33% | 83.61% | 0.8908 |
| Scikit-learn Linear SVM | 90.74% | 95.33% | 83.61% | 0.8908 |
| RBF SVM | **99.63%** | **99.19%** | **100%** | **0.9959** |

### Four-Feature Cross-Validation Results

Stratified 5-fold cross-validation was performed using the training data.

| Model | Validation Accuracy | Validation F1 |
|---|---:|---:|
| Manual Decision Tree | 93.51% | 0.9289 |
| Scikit-learn Decision Tree | 93.51% | 0.9289 |
| Manual Linear SVM | 86.46% | 0.8293 |
| Scikit-learn Linear SVM | 86.46% | 0.8293 |
| RBF SVM | **99.91%** | **0.9990** |

## 5. Key Observations

- Manual Decision Tree and Scikit-learn Decision Tree achieved identical classification metrics in the final comparison.
- Manual Linear SVM and Scikit-learn Linear SVM also produced identical classification metrics.
- Decision Trees provided interpretable feature-based splitting rules without requiring feature standardization.
- SVMs required feature standardization for effective optimization.
- The RBF kernel captured nonlinear relationships more effectively than the linear and polynomial kernels in the experiments.
- RBF SVM achieved the highest test and cross-validation performance.
- Scikit-learn implementations were considerably faster than the corresponding manual implementations due to optimized numerical algorithms.
- Increasing model complexity can increase the risk of overfitting, making validation essential.

## 6. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 7. How to Run

1. Clone the repository.
2. Install the required dependencies.
3. Open the Lab 08 Jupyter Notebook.
4. Ensure the Banknote Authentication dataset is available at the path specified in the notebook.
5. Execute all cells sequentially to reproduce the experiments, visualizations, and evaluation results.

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## 8. Conclusion

This lab provided practical experience in implementing Decision Trees and Linear SVMs from scratch and understanding their underlying mathematical operations.

Comparisons with Scikit-learn helped verify the manual implementations, while kernel-based experiments demonstrated the importance of nonlinear decision boundaries.

Among the evaluated models, the RBF SVM achieved the best classification performance on the Banknote Authentication dataset, while Decision Trees offered a more interpretable alternative.
