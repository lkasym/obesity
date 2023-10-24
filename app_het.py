import streamlit as st
import pandas as pd
from joblib import load

# Load pre-trained models
model_files = {
    "KNN": "KNN.pkl",
    "Logistic Regression": "Logistic Regression.pkl",
    "Naive Bayes": "Naive Bayes.pkl",
    "SVM": "SVM.pkl",
    "Random Forest": "Random Forest.pkl",
    "Decision Tree": "Decision Tree.pkl",
    "Gradient Boosting": "Gradient Boosting.pkl"
}

models = {}
for model_name, model_file in model_files.items():
    with open(model_file, 'rb') as f:
        models[model_name] = load(f)

# Streamlit interface
st.title("Obesity Prediction")

# Collect user input
age = st.number_input("Age", min_value=10, max_value=100)
height = st.number_input("Height (in meters)", min_value=1.0, max_value=2.5)
weight = st.number_input("Weight (in kgs)", min_value=30, max_value=200)
bmi = weight / (height ** 2)

# Encoding and collecting user input
gender = st.selectbox("Gender", ["Male", "Female"])
gender_encoded = 0 if gender == 'Male' else 1

family_history_with_overweight = st.selectbox("Family history with overweight?", ["Yes", "No"])
family_history_with_overweight_encoded = 1 if family_history_with_overweight == 'Yes' else 0

FAVC = st.selectbox("Frequent consumption of high caloric food?", ["Yes", "No"])
FAVC_encoded = 1 if FAVC == 'Yes' else 0

FCVC = st.selectbox("Frequency of vegetable consumption", ["Never", "Sometimes", "Always"])
FCVC_encoded = {"Never": 0, "Sometimes": 1, "Always": 2}[FCVC]

NCP = st.selectbox("Number of main meals a day", ["1 meal", "2 meals", "3 meals", "4 meals"])
NCP_encoded = {"1 meal": 0, "2 meals": 1, "3 meals": 2, "4 meals": 3}[NCP]

CAEC = st.selectbox("Consumption of food between meals", ["No", "Sometimes", "Frequently", "Always"])
CAEC_encoded = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[CAEC]

SMOKE = st.selectbox("Do you smoke?", ["Yes", "No"])
SMOKE_encoded = 1 if SMOKE == 'Yes' else 0

CH2O = st.selectbox("Daily water consumption", ["Less than a liter", "Between 1 and 2 liters", "More than 2 liters"])
CH2O_encoded = {"Less than a liter": 0, "Between 1 and 2 liters": 1, "More than 2 liters": 2}[CH2O]

SCC = st.selectbox("Do you monitor calorie consumption?", ["Yes", "No"])
SCC_encoded = 1 if SCC == 'Yes' else 0

FAF = st.selectbox("Frequency of physical activity", ["None", "1-2 days a week", "3-4 days a week", "4-5 days a week"])
FAF_encoded = {"None": 0, "1-2 days a week": 1, "3-4 days a week": 2, "4-5 days a week": 3}[FAF]

TUE = st.selectbox("Time using technology devices daily", ["0-2 hours", "3-5 hours", "More than 5 hours"])
TUE_encoded = {"0-2 hours": 0, "3-5 hours": 1, "More than 5 hours": 2}[TUE]

CALC = st.selectbox("Frequency of alcohol consumption", ["No", "Sometimes", "Frequently", "Always"])
CALC_encoded = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[CALC]

MTRANS = st.selectbox("MTRANS", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
MTRANS_encoded = {
    "Public_Transportation": 0,
    "Walking": 1,
    "Automobile": 2,
    "Motorbike": 3,
    "Bike": 4
}[MTRANS]

# Construct the encoded and preprocessed input data
input_data_encoded = {
    'Gender': gender_encoded,
    'Age': age,
    
    'family_history_with_overweight': family_history_with_overweight_encoded,
    'FAVC': FAVC_encoded,
    'FCVC': FCVC_encoded,
    'NCP': NCP_encoded,
    'CAEC': CAEC_encoded,
    'SMOKE': SMOKE_encoded,
    'CH2O': CH2O_encoded,
    'SCC': SCC_encoded,
    'FAF': FAF_encoded,
    'TUE': TUE_encoded,
    'CALC': CALC_encoded,
    'MTRANS': MTRANS_encoded,
    'BMI': bmi
}

input_df = pd.DataFrame([input_data_encoded])

# Model selection
selected_model = st.selectbox(
    'Choose a model for prediction:',
    list(models.keys())
)

# Prediction
if st.button("Predict"):
    model = models[selected_model]
    prediction = model.predict(input_df)
    st.write(f"Predicted Class: {prediction[0]}")
