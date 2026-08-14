# DS605 Lab 02 - NumPy, Pandas and Titanic Data Analysis

This repository contains the implementation for **DS605 Lab 02**. The lab covers fundamental NumPy operations, descriptive statistics, matrix operations, random sampling from a normal distribution, Pandas DataFrame operations, Boolean filtering, grouping and aggregation, missing-value imputation, outlier detection, feature engineering, pivot tables, correlation analysis, and basic visualization using the Titanic dataset.

## Contents

- `202618002_DS605_LAB02 (1).ipynb` - Jupyter Notebook containing all lab tasks and code
- `train.csv` - Titanic training dataset used in Part B

## Requirements

The notebook uses Python and the following libraries:

```text
numpy
pandas
matplotlib
seaborn
```

Install them with:

```bash
pip install numpy pandas matplotlib seaborn
```

## Lab Structure

### Part A - NumPy and Statistics

#### Task 1 - Arrays, Statistics, and Indexing

The notebook:

- Generates an array `A` containing 100 random integers.
- Sets a random seed for reproducibility.
- Calculates the minimum, maximum, median, mean, and standard deviation of `A`.
- Generates an array `B` containing exactly 100 values using `np.arange()`.
- Demonstrates `np.zeros()` and `np.ones()`, including their shapes and data types.
- Demonstrates `np.linspace()` and compares it with `np.arange()`.
- Creates 2D and 3D arrays.
- Demonstrates shape, number of dimensions, indexing, rows, columns, depth, and slicing.
- Uses `reshape()` and `flatten()` to reshape and convert arrays.

#### Task 2 - Matrix Operations

Two matrices are created and the following operations are performed:

- Matrix addition
- Element-wise multiplication
- Matrix multiplication using `@`
- Matrix transpose
- Determinant calculation
- Matrix inverse when the matrix is invertible
- Invertibility checking

NumPy vectorized operations are used instead of explicit Python loops.

#### Normal Distribution and Sampling

The notebook generates **1,000 values from a normal distribution** using:

- Mean = `100`
- Standard deviation = `15`
- Number of observations = `1000`

The sample mean and sample standard deviation are calculated and compared with the chosen distribution parameters.

A histogram is also plotted to visualize the generated normal distribution.

---

# Part B - Titanic Dataset Analysis

The Titanic `train.csv` dataset is used for the Pandas-based tasks.

## Task 4 - DataFrame Inspection and Indexing

The notebook demonstrates:

- `head()`
- `tail()`
- `shape`
- `columns`
- `info()`
- `describe()`

It also demonstrates the difference between:

- `loc` - label-based indexing
- `iloc` - integer-position-based indexing

Example:

```python
df.loc[4, ["Name", "Sex"]]
df.iloc[4, [3, 4]]
```

`loc` selects using row/column labels, while `iloc` selects using integer positions.

## Task 5 - Boolean Indexing and `query()`

The notebook uses Boolean conditions and `query()` to answer questions about Titanic passengers, including:

- Number of male passengers older than 50
- Number of female first-class passengers
- Percentage of female first-class passengers who survived
- Number of passengers aged 20-40 with above-median fare who survived
- Number of passengers travelling alone, younger than 30, who did not survive
- Number of Southampton passengers in Pclass 2 or 3 whose fare was above the Southampton median

## Task 6 - Grouping and Aggregation

The following grouped statistics are calculated:

- Survival rate by sex
- Survival rate by passenger class
- Average age and fare by passenger class
- Passenger count and survival rate by sex and passenger class
- Passenger count, average fare, and survival rate by embarkation location

Examples of Pandas operations used include:

```python
groupby()
agg()
mean()
count()
```

## Task 7 - Missing Values and Outliers

The notebook:

- Calculates missing-value counts for every column.
- Calculates missing-value percentages.
- Plots missing-value counts using a bar chart.
- Fills missing `Age` values using mean imputation.
- Compares multiple imputation methods:
  - Mean
  - Median
  - Mode
  - Random value drawn from observed `Age` values
- Calculates Fare:
  - Q1
  - Q3
  - IQR
  - Lower and upper 1.5 × IQR bounds
  - Number of outliers

The standard IQR rule used is:

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

## Task 8 - Feature Engineering and Pivot Table

Two new features are created:

### FamilySize

```python
FamilySize = SibSp + Parch + 1
```

### IsAlone

`IsAlone` is set to `1` when `FamilySize == 1`, otherwise `0`.

A pivot table is then created with:

- Rows: `Sex`
- Columns: `Pclass`
- Values: mean `Survived`

This is used to compare survival rates across sex and passenger class groups.

## Task 9 - Correlation and Visualization

The notebook includes:

### Correlation Heatmap

A correlation heatmap is created for:

```text
Age
Fare
Pclass
SibSp
Parch
Survived
FamilySize
IsAlone
```

The heatmap is used to identify positive and negative relationships between numerical variables.

Some notable correlations from the analysis include:

| Variable Pair | Correlation |
|---|---:|
| SibSp - FamilySize | 0.89 |
| Parch - FamilySize | 0.78 |
| FamilySize - IsAlone | -0.69 |
| Fare - Pclass | -0.55 |
| SibSp - IsAlone | -0.58 |
| Parch - IsAlone | -0.58 |
| Pclass - Survived | -0.34 |
| Fare - Survived | 0.26 |
| Age - Survived | -0.08 |

The strong relationships involving `FamilySize` are expected because `FamilySize` is constructed from `SibSp` and `Parch`.

### Survival Rate by Sex

A bar chart is used to compare male and female survival rates.

The analysis shows substantially higher survival among female passengers than male passengers.

### Age vs Fare

A scatter plot of `Age` versus `Fare` is created, with points distinguished according to whether the passenger survived.

---

# Key Observations

1. `SibSp` and `FamilySize`, as well as `Parch` and `FamilySize`, have strong positive correlations because `FamilySize` is constructed using these variables.
2. `FamilySize` and `IsAlone` have a strong negative relationship because passengers with larger family sizes are less likely to be classified as travelling alone.
3. `Fare` and `Pclass` have a moderately strong negative correlation. This is consistent with the coding of passenger classes, where `1` represents first class and `3` represents third class.
4. Female passengers had substantially higher survival rates than male passengers.
5. `Age` has a very weak linear correlation with `Survived`, with a correlation of approximately `-0.08`.
6. `Age` has a moderate negative correlation with `Pclass` (`-0.37`), while `Age` and `SibSp` have a moderate negative correlation (`-0.31`).
7. Most Titanic passengers paid relatively low fares, and the scatter plot shows a dense concentration of observations in the lower-fare range.

## Important Interpretation Note

Correlation measures the strength and direction of a **linear association**. It does not by itself establish causation.

For example, a correlation between `Fare` and `Survived` does not prove that paying a higher fare directly caused a passenger to survive. Passenger class and other associated factors may also contribute to the observed relationship.

## Running the Notebook

1. Clone or download this repository.
2. Place `train.csv` in the expected dataset location.
3. Open the notebook in Jupyter Notebook, JupyterLab, Google Colab, or VS Code.
4. Update the `pd.read_csv()` path if the location of `train.csv` differs from the path used in the notebook.
5. Run the cells sequentially.

The current notebook loads the Titanic dataset using:

```python
df = pd.read_csv("/content/drive/MyDrive/DAU/MACHINE LEARNING/LAB02/train.csv")
```

If running outside Google Colab, this path should be changed to the local location of `train.csv`, for example:

```python
df = pd.read_csv("train.csv")
```

## Tools and Libraries

- **Python** - Programming language
- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization and correlation heatmap

## Author

**202618002**

DS605 - Data Science Lab 02
