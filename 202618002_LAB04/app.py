# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import math
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Define the custom wrapper class so pickle can find it
class FillNaWrapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): 
        return self
    def transform(self, X): 
        return X.fillna('')

# 2. Load the saved preprocessor and model
model = joblib.load('best_airbnb_model_grid.pkl')
preprocessor = joblib.load('preprocessor_pipeline.pkl')

st.set_page_config(page_title="NYC Airbnb Price Predictor", page_icon="🏙️", layout="centered")

st.title("🏙️ NYC Airbnb Price Predictor")
st.write("Provide the details of the listing below to get an instant AI-powered prediction for the optimal nightly price.")

st.divider()

# Input Form layout
with st.form("prediction_form"):
    st.subheader("Listing Information")
    
    name = st.text_input("Listing Title / Name", "Cozy modern loft in Manhattan")
    
    col1, col2 = st.columns(2)
    with col1:
        neighbourhood_group = st.selectbox("Borough (Neighbourhood Group)", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
        neighbourhood = st.text_input("Specific Neighbourhood", "Midtown")
        room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
        latitude = st.number_input("Latitude", value=40.7580, format="%.5f")
        longitude = st.number_input("Longitude", value=-73.9855, format="%.5f")
        
    with col2:
        minimum_nights = st.number_input("Minimum Nights", value=2, min_value=1, max_value=365)
        availability_365 = st.number_input("Availability (Days out of 365)", value=180, min_value=0, max_value=365)
        number_of_reviews = st.number_input("Total Number of Reviews", value=15, min_value=0)
        reviews_per_month = st.number_input("Reviews Per Month", value=1.0, min_value=0.0)
        host_listings_count = st.number_input("Host Total Listings Count", value=1, min_value=1)
        days_since_last_review = st.number_input("Days Since Last Review", value=30.0, min_value=0.0)

    submit_button = st.form_submit_button(label="Predict Price")

if submit_button:
    # 1. Re-apply feature engineering steps required by the pipeline
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    manhattan_center = (40.7580, -73.9855)
    jfk_airport = (40.6413, -73.7781)
    
    distance_to_center = haversine(latitude, longitude, manhattan_center[0], manhattan_center[1])
    distance_to_jfk = haversine(latitude, longitude, jfk_airport[0], jfk_airport[1])
    
    host_type = 'Multi-listing' if host_listings_count > 1 else 'Single-listing'
    
    review_rate = 0.50
    est_occupancy_days_per_month = min(max((reviews_per_month / review_rate) * minimum_nights, 0), 30)
    
    if availability_365 == 0:
        availability_bucket = 'Zero'
    elif availability_365 <= 90:
        availability_bucket = 'Low'
    elif availability_365 <= 200:
        availability_bucket = 'Medium'
    else:
        availability_bucket = 'High'
        
    if minimum_nights <= 3:
        minimum_nights_bucket = 'short'
    elif minimum_nights <= 14:
        minimum_nights_bucket = 'medium'
    else:
        minimum_nights_bucket = 'long'
        
    room_type_neighbourhood = f"{room_type}_{neighbourhood_group}"

    # 2. Build input dataframe matching training schema
    input_df = pd.DataFrame([{
        'name': name,
        'neighbourhood_group': neighbourhood_group,
        'neighbourhood': neighbourhood,
        'latitude': latitude,
        'longitude': longitude,
        'room_type': room_type,
        'minimum_nights': minimum_nights,
        'number_of_reviews': number_of_reviews,
        'reviews_per_month': reviews_per_month,
        'availability_365': availability_365,
        'host_type': host_type,
        'distance_to_center': distance_to_center,
        'distance_to_jfk': distance_to_jfk,
        'est_occupancy_days_per_month': est_occupancy_days_per_month,
        'availability_bucket': availability_bucket,
        'minimum_nights_bucket': minimum_nights_bucket,
        'room_type_neighbourhood': room_type_neighbourhood,
        'days_since_last_review': days_since_last_review
    }])

    # 3. Transform inputs through saved pipeline and predict
    processed_input = preprocessor.transform(input_df)
    log_prediction = model.predict(processed_input)
    final_price = np.expm1(log_prediction)[0]

    st.success(f"### Estimated Nightly Price: ${final_price:.2f}")