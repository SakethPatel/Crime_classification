import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset (replace with actual path to the dataset)
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

train_data = load_data('train.csv')

# Extract mappings dynamically
status_mapping = dict(zip(train_data['Status'].unique(), train_data['Status_Description'].unique()))
premise_mapping = dict(zip(train_data['Premise_Code'].unique(), train_data['Premise_Description'].unique()))
weapon_mapping = dict(zip(train_data['Weapon_Description'].unique(), train_data['Weapon_Description'].unique()))

# Data Preprocessing and Feature Engineering
train_data['Date_Occurred'] = pd.to_datetime(train_data['Date_Occurred'])
train_data['Date_Reported'] = pd.to_datetime(train_data['Date_Reported'])
train_data['Reported-Occured'] = (train_data['Date_Reported'] - train_data['Date_Occurred']).dt.days

# Time bins for classification
train_data['Time_Occurred'] = pd.to_datetime(train_data['Time_Occurred'], format='%H%M', errors='coerce')
bins = [0, 6, 12, 18, 24]
labels = ['Night', 'Morning', 'Afternoon', 'Evening']
train_data['Time_Occurred_Label'] = pd.cut(train_data['Time_Occurred'].dt.hour.fillna(0), bins=bins, labels=labels, right=False)

# Reported bins calculation
def calculate_reported_bin(date_occurred, date_reported):
    days_diff = (date_reported - date_occurred).days
    if days_diff <= 15:
        return "Within 15 days"
    elif days_diff <= 180:
        return "15 days to 6 months"
    elif days_diff <= 365:
        return "6 months to 1 year"
    else:
        return "Greater than 1 year"

train_data['Reported_bins'] = train_data.apply(lambda row: calculate_reported_bin(row['Date_Occurred'], row['Date_Reported']), axis=1)

# Selecting relevant columns
X = train_data.drop(columns=['Crime_Category', 'Date_Occurred', 'Date_Reported'])
y = train_data['Crime_Category']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encoding and scaling
encoder = ce.TargetEncoder(cols=['Location', 'Victim_Sex', 'Weapon_Description', 'Premise_Code', 'Victim_Descent', 'Status', 'Reporting_District_no', 'Time_Occurred_Label', 'Reported_bins'])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and encoders
joblib.dump(model, 'crime_classifier_model.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Predictions
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit App
st.title("Crime Classification Application")

# Display model accuracy
st.write(f"Model Accuracy: {accuracy:.4f}")

# Status dropdown
status = st.selectbox('Select the Crime Status:', list(status_mapping.keys()), help="Select the status of the crime as reported.")
st.write(f"Status Description: {status_mapping.get(status)}")

# Premise Code dropdown
premise_code = st.selectbox('Select the Premise Code:', list(premise_mapping.keys()), help="Select the premise where the crime occurred.")
st.write(f"Premise Description: {premise_mapping.get(premise_code)}")

# Weapon Description dropdown
weapon_description = st.selectbox('Select the Weapon Description:', list(weapon_mapping.keys()), help="Select the weapon used in the crime, if applicable.")
st.write(f"Weapon Description: {weapon_description}")

# Date input
date_occurred = st.date_input('Select the Crime Occurrence Date:', help="Date when the crime occurred.")
date_reported = st.date_input('Select the Crime Reported Date:', help="Date when the crime was reported.")

if date_occurred and date_reported:
    if date_occurred > date_reported:
        st.error("Error: 'Date Reported' must be on or after 'Date Occurred'. Please correct the input.")
    else:
        reported_bin = calculate_reported_bin(date_occurred, date_reported)
        st.write(f"Reported Bin: {reported_bin}")

# User inputs for prediction
location = st.text_input("Enter Crime Location:")
victim_sex = st.selectbox("Select Victim Sex:", ['Male', 'Female', 'Unknown'])
victim_descent = st.selectbox("Select Victim Descent:", ['White', 'Black', 'Asian', 'Other', 'Unknown'])
time_occurred = st.selectbox("Select Time of Crime:", ['Night', 'Morning', 'Afternoon', 'Evening'])

if location and reported_bin:
    # Load saved models
    model = joblib.load('crime_classifier_model.pkl')
    encoder = joblib.load('encoder.pkl')
    scaler = joblib.load('scaler.pkl')

    # Construct the feature vector for prediction
    user_input = pd.DataFrame({
        'Location': [location],
        'Victim_Sex': [victim_sex],
        'Weapon_Description': [weapon_description],
        'Premise_Code': [premise_code],
        'Victim_Descent': [victim_descent],
        'Time_Occurred_Label': [time_occurred],
        'Reported_bins': [reported_bin]
    })

    user_input_encoded = encoder.transform(user_input)
    user_input_scaled = scaler.transform(user_input_encoded)

    # Make a prediction
    prediction = model.predict(user_input_scaled)

    # Show the predicted crime category
    st.write(f"Predicted Crime Category: {prediction[0]}")
