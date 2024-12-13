import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import category_encoders as ce
from sklearn.metrics import accuracy_score
import time

# Load the dataset (replace with actual path to the dataset)
train_data = pd.read_csv('train.csv')

# Extract the Status to Status Description mapping dynamically
status_mapping = dict(zip(train_data['Status'].unique(), train_data['Status_Description'].unique()))

# Extract the Premise Code to Premise Description mapping dynamically
premise_mapping = dict(zip(train_data['Premise_Code'].unique(), train_data['Premise_Description'].unique()))

# Extract Weapon Description mapping dynamically (if available in the dataset)
weapon_mapping = dict(zip(train_data['Weapon_Description'].unique(), train_data['Weapon_Description'].unique()))

# Data Preprocessing and Feature Engineering
train_data['Date_Occurred'] = pd.to_datetime(train_data['Date_Occurred'])
train_data['Date_Reported'] = pd.to_datetime(train_data['Date_Reported'])
train_data['Reported-Occured'] = (train_data['Date_Reported'] - train_data['Date_Occurred']).dt.days

# Time and bins for classification
train_data['Time_Occurred'] = pd.to_datetime(train_data['Time_Occurred'], format='%H%M')
bins = [0, 6, 12, 18, 24]
labels = ['Night', 'Morning', 'Afternoon', 'Evening']
train_data['Time_Occurred_Label'] = pd.cut(train_data['Time_Occurred'].dt.hour, bins=bins, labels=labels, right=False)

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

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply OneHotEncoder and StandardScaler
encoder = ce.TargetEncoder(cols=['Location', 'Victim_Sex', 'Weapon_Description', 'Premise_Code', 'Victim_Descent', 'Status', 'Reporting_District_no', 'Time_Occurred_Label', 'Reported_bins'])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit App UI
st.title("Crime Classification Application")

# Display Model Accuracy
st.write(f"Model Accuracy: {accuracy:.4f}")

# Status dropdown (user selects Status)
status = st.selectbox('Select the Crime Status:', status_mapping.keys())
st.write(f"Status Description: {status_mapping.get(status)}")

# Premise Code dropdown (user selects Premise Code)
premise_code = st.selectbox('Select the Premise Code:', premise_mapping.keys())
st.write(f"Premise Description: {premise_mapping.get(premise_code)}")

# Weapon Description dropdown (user selects Weapon Description)
weapon_description = st.selectbox('Select the Weapon Description:', weapon_mapping.keys())
st.write(f"Weapon Description: {weapon_mapping.get(weapon_description)}")

# Date input for 'Date Occurred' and 'Date Reported'
date_occurred = st.date_input('Select the Crime Occurrence Date:')
date_reported = st.date_input('Select the Crime Reported Date:')

# Calculate the Reported Bin dynamically
if date_occurred and date_reported:
    reported_bin = calculate_reported_bin(date_occurred, date_reported)
    st.write(f"Reported Bin: {reported_bin}")

# User Inputs for Prediction
location = st.text_input("Enter Crime Location:")
victim_sex = st.selectbox("Select Victim Sex:", ['Male', 'Female', 'Unknown'])
victim_descent = st.selectbox("Select Victim Descent:", ['White', 'Black', 'Asian', 'Other', 'Unknown'])
time_occurred = st.selectbox("Select Time of Crime:", ['Night', 'Morning', 'Afternoon', 'Evening'])

# Constructing the feature vector for prediction based on the user input
user_input = pd.DataFrame({
    'Location': [location],
    'Victim_Sex': [victim_sex],
    'Weapon_Description': [weapon_description],
    'Premise_Code': [premise_code],
    'Victim_Descent': [victim_descent],
    'Time_Occurred_Label': [time_occurred],
    'Reported_bins': [reported_bin]
})

# Transform the user input using the same encoders used during training
user_input_encoded = encoder.transform(user_input)
user_input_scaled = scaler.transform(user_input_encoded)

# Make a prediction
prediction = model.predict(user_input_scaled)

# Show the predicted crime category
st.write(f"Predicted Crime Category: {prediction[0]}")
