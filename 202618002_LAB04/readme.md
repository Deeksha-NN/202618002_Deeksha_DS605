# NYC Airbnb Price Prediction Platform

A machine learning system that estimates optimal nightly rental prices for short-term properties across New York City's five boroughs.

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

## Installation & Running Locally

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   
## Some demonstrations:

<img width="799" height="875" alt="image" src="https://github.com/user-attachments/assets/d7b78afa-3b4f-4251-b47e-d06cf486dccd" />

<img width="808" height="865" alt="image" src="https://github.com/user-attachments/assets/24d54e6b-0b79-42f8-95b9-6e9c477cf23f" />

