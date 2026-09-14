# NYC Airbnb Price Prediction Platform

Predicting short-term rental pricing in a dynamic market like New York City presents unique challenges due to sparse text descriptions, wide spatial variations, and extreme luxury outliers. The primary objective was to build a robust regression pipeline that generalizes well across everyday listings while providing an accessible evaluation tool for prospective hosts.

A machine learning system that estimates optimal nightly rental prices for properties across New York City's five boroughs.

## Features
- **Geospatial & Text Features**: Incorporates Haversine distance calculations and TF-IDF text analysis of listing titles.
- **Optimized XGBoost Backend**: Powered by a grid-searched XGBoost regressor trained on log-transformed market data.
- **Interactive Web UI**: Built with Streamlit for seamless user inputs and real-time inference.

## Project Directory Structure
```text
├── AB_NYC_2019.csv            # Raw NYC Airbnb dataset
├── Airbnb_task1.ipynb         # Task 1: Data cleaning, feature engineering & preprocessing pipeline
├── AIRBNB_TASK2.ipynb         # Task 2: Model training, evaluation, and GridSearchCV optimization
├── X_train_processed.pkl      # Processed training features array
├── X_test_processed.pkl       # Processed testing features array
├── y_train.pkl                # Training target values (log space)
├── y_test.pkl                 # Testing target values (log space)
├── preprocessor_pipeline.pkl  # Serialized scikit-learn preprocessing pipeline (with custom wrapper)
├── best_airbnb_model_grid.pkl # Serialized grid-tuned XGBoost model
├── app.py                     # Streamlit web application for real-time inference
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation

Baseline Comparison:

Tree-based models (XGBoost, Random Forest, LightGBM) drastically outperformed linear models by capturing non-linear geographic boundaries and interaction effects.

Evaluation Metrics:
$R^2$ Score (0.63): The final grid-tuned XGBoost model explains approximately 63% of the variance in NYC listing prices.
MAE ($53.21): For standard everyday listings, predictions typically miss the actual market price by roughly $53, representing strong real-world utility.
RMSE ($199.60): Reflected the disproportionate mathematical penalty inflicted by extreme multi-thousand-dollar luxury penthouses and townhomes present in the tail of the dataset.

## Installation & Running Locally

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   
## Some demonstrations:

<img width="799" height="875" alt="image" src="https://github.com/user-attachments/assets/d7b78afa-3b4f-4251-b47e-d06cf486dccd" />

<img width="808" height="865" alt="image" src="https://github.com/user-attachments/assets/24d54e6b-0b79-42f8-95b9-6e9c477cf23f" />

