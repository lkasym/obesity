import streamlit as st
import pandas as pd
import pickle
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
        models[model_name] = load(f'{model_name}.pkl')

# Define a list of feature names in the correct order
feature_names = [
    'Age', 'Gender', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE',
    'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'BMI'
]

# Create a dictionary for mapping user inputs to numerical values
input_mapping = {
    'Gender': {'Male': 0, 'Female': 1},
    'family_history_with_overweight': {'Yes': 1, 'No': 0},
    'FAVC': {'Yes': 1, 'No': 0},
    'FCVC': {'Never': 0, 'Sometimes': 1, 'Always': 2},
    'NCP': {'1 meal': 0, '2 meals': 1, '3 meals': 2, '4 meals': 3},
    'CAEC': {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'SMOKE': {'Yes': 1, 'No': 0},
    'CH2O': {'Less than a liter': 0, 'Between 1 and 2 liters': 1, 'More than 2 liters': 2},
    'SCC': {'Yes': 1, 'No': 0},
    'FAF': {'None': 0, '1-2 days a week': 1, '3-4 days a week': 2, '4-5 days a week': 3},
    'TUE': {'0-2 hours': 0, '3-5 hours': 1, 'More than 5 hours': 2},
    'CALC': {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
}

# Streamlit interface
st.title("Obesity Prediction")

# Collect user input
age = st.number_input("Age", min_value=10, max_value=100)
gender = st.selectbox("Gender", ["Male", "Female"])
family_history = st.selectbox("Family history with overweight?", ["Yes", "No"])
favc = st.selectbox("Frequent consumption of high caloric food?", ["Yes", "No"])
fcvc = st.selectbox("Frequency of vegetable consumption", ["Never", "Sometimes", "Always"])
ncp = st.selectbox("Number of main meals a day", ["1 meal", "2 meals", "3 meals", "4 meals"])
caec = st.selectbox("Consumption of food between meals", ["No", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Do you smoke?", ["Yes", "No"])
ch2o = st.selectbox("Daily water consumption", ["Less than a liter", "Between 1 and 2 liters", "More than 2 liters"])
scc = st.selectbox("Do you monitor calorie consumption?", ["Yes", "No"])
faf = st.selectbox("Frequency of physical activity", ["None", "1-2 days a week", "3-4 days a week", "4-5 days a week"])
tue = st.selectbox("Time using technology devices daily", ["0-2 hours", "3-5 hours", "More than 5 hours"])
calc = st.selectbox("Frequency of alcohol consumption", ["No", "Sometimes", "Frequently", "Always"])
height = st.number_input("Height (in meters)", min_value=1.0, max_value=2.5)
weight = st.number_input("Weight (in kgs)", min_value=30, max_value=200)
bmi = weight / (height ** 2)

# Map user inputs to numerical values based on the input_mapping dictionary
input_data = {
    'Age': age,
    'Gender': gender,
    'family_history_with_overweight': family_history,
    'FAVC': favc,
    'FCVC': fcvc,
    'NCP': ncp,
    'CAEC': caec,
    'SMOKE': smoke,
    'CH2O': ch2o,
    'SCC': scc,
    'FAF': faf,
    'TUE': tue,
    'CALC': calc,
    'MTRANS': 'Public_Transportation',  # Default value, you can change this as needed
    'BMI': bmi
}

# Create the input DataFrame with the correct order of features
input_df = pd.DataFrame([input_data], columns=feature_names)

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
