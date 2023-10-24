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
        
        


# Streamlit interface
st.title("Obesity Prediction")

# Collect user input
age = st.number_input("Age", min_value=10, max_value=100)
height = st.number_input("Height (in meters)", min_value=1.0, max_value=2.5)
weight = st.number_input("Weight (in kgs)", min_value=30, max_value=200)
bmi = weight / (height ** 2)

input_data = {
    'Age': age,
    
    'family_history_with_overweight': st.selectbox("Family history with overweight?", ["Yes", "No"]),
    'FAVC': st.selectbox("Frequent consumption of high caloric food?", ["Yes", "No"]),
    'FCVC': st.selectbox("Frequency of vegetable consumption", ["Never", "Sometimes", "Always"]),
    'NCP': st.selectbox("Number of main meals a day", ["1 meal", "2 meals", "3 meals", "4 meals"]),
    'CAEC': st.selectbox("Consumption of food between meals", ["No", "Sometimes", "Frequently", "Always"]),
    'SMOKE': st.selectbox("Do you smoke?", ["Yes", "No"]),
    'CH2O': st.selectbox("Daily water consumption", ["Less than a liter", "Between 1 and 2 liters", "More than 2 liters"]),
    'SCC': st.selectbox("Do you monitor calorie consumption?", ["Yes", "No"]),
    'FAF': st.selectbox("Frequency of physical activity", ["None", "1-2 days a week", "3-4 days a week", "4-5 days a week"]),
    'TUE': st.selectbox("Time using technology devices daily", ["0-2 hours", "3-5 hours", "More than 5 hours"]),
    'CALC': st.selectbox("Frequency of alcohol consumption", ["No", "Sometimes", "Frequently", "Always"]),
    'BMI': bmi
}

input_df = pd.DataFrame([input_data])

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
