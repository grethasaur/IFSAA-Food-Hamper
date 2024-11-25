
import pandas as pd
import joblib 
import gradio as gr
import numpy as np
from datetime import datetime
from scipy.stats import boxcox
from scipy.special import inv_boxcox

# Assuming historical_data and model_rf are already defined
# Load the model and historical data (assuming historical data is loaded in a dataframe)
final_model, fitted_lambda = joblib.load("trained_model_and_lambda.pkl")

# Assuming historical_data has been pre-processed and contains lagged features
historical_data = pd.read_csv('historical_data.csv')

historical_data['date'] = pd.to_datetime(historical_data['date']).dt.date

# Set the date column as index
historical_data.set_index('date', inplace=True)


# Function to calculate lagged features (same logic as training)
def create_lagged_features_for_future(historical_data, input_date):
    # Get the most recent date in the historical data (using index)
    last_available_date = historical_data.index[-1]

    # Check if input date is after the last available date
    if input_date <= last_available_date:
        return "Error: Input date must be after the most recent historical data date."
    
    # Generate lagged features based on the most recent data
    lagged_features = {}

    # Add the current 'scheduled_date_count' value (this is the missing feature)
    lagged_features['scheduled_date_count'] = historical_data.loc[last_available_date, 'scheduled_date_count']
    
    for lag in [7, 14, 21]:
        # For each lag, look at the last available data point (most recent date)
        lagged_features[f'pickup_date_count_lag_{lag}'] = historical_data.loc[last_available_date, f'pickup_date_count_lag_{lag}']
        lagged_features[f'scheduled_date_count_{lag}'] = historical_data.loc[last_available_date, f'scheduled_date_count_{lag}']

    return lagged_features


# Function to make predictions based on user input
def predict(year, month, day, scheduled_date_count):
    try:
        # Ensure the inputs are integers where necessary
        year = int(year)
        month = int(month)
        day = int(day)
        scheduled_date_count = int(scheduled_date_count)

        # Construct the input date as a datetime object
        input_date = datetime(year, month, day)

        # Generate lagged features for the input date
        features = calculate_lagged_features(input_date, scheduled_date_count)

        # Convert features to numpy array for prediction
        features_array = np.array([list(features.values())]).reshape(1, -1)
        
        # Make the prediction using the pre-trained model
        prediction = final_model.predict(features_array)

        # Reverse the Box-Cox transformation
        predicted_value = inv_boxcox(prediction[0], fitted_lambda) - 1

        # Round the prediction if necessary
        rounded_prediction = round(predicted_value)

        return f"Predicted Pickup Date Count for {input_date.strftime('%Y-%m-%d')}: {rounded_prediction}"

    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Define Gradio components
year_dropdown = gr.Dropdown(choices=[str(y) for y in range(2024, 2026)], label="Year", value="2025")
month_dropdown = gr.Dropdown(choices=[str(m).zfill(2) for m in range(1, 13)], label="Month", value="04")
day_dropdown = gr.Dropdown(choices=[str(d).zfill(2) for d in range(1, 32)], label="Day", value="11")
scheduled_count_input = gr.Number(label="Enter Scheduled Date Count", value=30)

output = gr.Textbox(label="Predicted Pickup Date Count")

# Create Gradio interface
app = gr.Interface(fn=predict, inputs=[year_dropdown, month_dropdown, day_dropdown, scheduled_count_input], outputs=output,
                   title="Pickup Date Count Predictor",
                   description="Select a date and enter the scheduled date count to predict the pickup date count.")

# Launch the app
if __name__ == "__main__":
    app.launch()

        
