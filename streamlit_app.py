import pandas as pd
import os
import requests
from io import BytesIO
import tempfile
import joblib
import numpy as np
import streamlit as st
from datetime import datetime
from scipy.special import inv_boxcox
from snowflake.snowpark.session import Session

# Establish Snowflake session
@st.cache_resource
def create_session():
    return Session.builder.configs(st.secrets["snowflake"]).create()

session = create_session()

# Test connection
st.success("Connected to Snowflake!")


# Load model from GitHub
@st.cache_resource
def load_model_from_github():
    model_url = "https://raw.githubusercontent.com/grethasaur/IFSAA-Food-Hamper/blob/ada46c96a8f697d2081f5af8e4b2b38658f62677/trained_model_and_lambda.pkl"
    # Fetch the model file from the GitHub raw URL
    response = requests.get(model_url)
    
    if response.status_code == 200:
        # Load the model from the byte content of the response
        model_data = BytesIO(response.content)
        model, fitted_lambda = joblib.load(model_data)
        return model, fitted_lambda
    else:
        st.error("Failed to load model from GitHub. Please check the URL or try again later.")
        return None, None

# Retrieve the historical data CSV from Snowflake stage
@st.cache_resource
def load_historical_data_from_snowflake():
    # Use the existing session object to query the table directly
    df = session.table("LAB.PUBLIC.HISTORICAL_DATA").to_pandas()

    # Check available columns to ensure 'scheduled_date' exists
    st.write(df.columns)  # Display the columns in the dataframe for debugging

    # Ensure the 'scheduled_date' column is in datetime format, adjust the name if needed
    if 'scheduled_date' in df.columns:
        df['scheduled_date'] = pd.to_datetime(df['scheduled_date']).dt.date
        df.set_index('scheduled_date', inplace=True)
    else:
        st.error("Column 'scheduled_date' not found in historical data!")
    
    return df
    
    return df

# Load resources
final_model, fitted_lambda = load_model_from_github()
historical_data = load_historical_data_from_snowflake()

# Function to calculate lagged features
def create_lagged_features(historical_data, input_date, scheduled_date_count):
    data = historical_data.copy()
    input_date = input_date.date()  # Ensure input_date is in datetime.date format

    new_row = pd.DataFrame({'scheduled_date_count': [scheduled_date_count]},
                           index=[input_date])
    data = pd.concat([data, new_row]).sort_index()

    for lag in [7, 14, 21]:
        data[f'pickup_date_count_lag_{lag}'] = data['scheduled_date_count'].shift(lag)
        data[f'scheduled_date_count_{lag}'] = data['scheduled_date_count'].shift(lag)

    lagged_features = data.loc[input_date]
    if lagged_features.isnull().any():
        raise ValueError("Insufficient historical data to calculate lagged features.")

    return lagged_features.to_dict()

# Function to make predictions
def predict(year, month, day, scheduled_date_count):
    try:
        year, month, day = int(year), int(month), int(day)
        scheduled_date_count = int(scheduled_date_count)
        input_date = datetime(year, month, day)

        features = create_lagged_features(historical_data, input_date, scheduled_date_count)
        features_array = np.array([list(features.values())]).reshape(1, -1)

        prediction = final_model.predict(features_array)
        predicted_value = inv_boxcox(prediction[0], fitted_lambda) - 1
        return round(predicted_value)
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit App
st.title("Pickup Date Count Predictor (Streamlit + Snowflake)")
st.write("Select a date and enter the scheduled date count to predict the pickup date count.")

# User inputs
year = st.selectbox("Year", [str(y) for y in range(2024, 2026)], index=1)
month = st.selectbox("Month", [str(m).zfill(2) for m in range(1, 13)], index=3)
day = st.selectbox("Day", [str(d).zfill(2) for d in range(1, 32)], index=10)
scheduled_date_count = st.number_input("Enter Scheduled Date Count", min_value=0, step=1, value=30)

# Prediction button
if st.button("Predict"):
    result = predict(year, month, day, scheduled_date_count)
    st.write(f"Predicted Pickup Count for {year}-{month}-{day} is: **{result} hampers**")
