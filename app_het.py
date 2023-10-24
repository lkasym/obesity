import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Streamlit App
st.title("Obesity Prediction")

# Collecting user inputs
age = st.number_input("Age", min_value=10, max_value=100)
height = st.number_input("Height (in meters)", min_value=1.0, max_value=2.5)
weight = st.number_input("Weight (in kgs)", min_value=30, max_value=200)
family_history_with_overweight = st.selectbox("Family history with overweight?", ["yes", "no"])
FAVC = st.selectbox("Frequent consumption of high caloric food?", ["yes", "no"])
FCVC = st.slider("Frequency of vegetable consumption", 1, 3)
NCP = st.slider("Number of main meals a day", 1, 4)
CAEC = st.selectbox("Consumption of food between meals", ["No", "Sometimes", "Frequently", "Always"])
SMOKE = st.selectbox("Do you smoke?", ["yes", "no"])
CH2O = st.slider("Daily water consumption", 1, 3)
SCC = st.selectbox("Do you monitor calorie consumption?", ["yes", "no"])
FAF = st.slider("Frequency of physical activity", 0, 3)
TUE = st.slider("Time using technology devices daily", 0, 2)
CALC = st.selectbox("Frequency of alcohol consumption", ["No", "Sometimes", "Frequently", "Always"])

# Dropdown to select model for prediction
model_option = st.selectbox(
    'Choose a model for prediction:',
    ('KNN', 'Logistic Regression', 'Naive Bayes', 'SVM', 'Random Forest', 'Decision Tree', 'Gradient Boosting')
)

# Button to predict
if st.button("Predict"):
    # Prepare the input data in the same format as your training data
    input_data = [age, height, weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC]
    input_df = pd.DataFrame([input_data], columns=['Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC'])
    
    # Load the provided dataset
    dataset = pd.read_csv("C:\\Users\\laksh\\Downloads\\preprocessed_obesity_dataset_updated.csv")
    X = dataset.drop('NObeyesdad', axis=1)
    y = dataset['NObeyesdad']
    
    # Predict using the chosen model
    prediction = None
    if model_option == 'KNN':
        knn_model = KNeighborsClassifier().fit(X, y)
        prediction = knn_model.predict(input_df)

    elif model_option == 'Logistic Regression':
        lr_model = LogisticRegression(max_iter=1000).fit(X, y)
        prediction = lr_model.predict(input_df)

    elif model_option == 'Naive Bayes':
        nb_model = GaussianNB().fit(X, y)
        prediction = nb_model.predict(input_df)

    elif model_option == 'SVM':
        svm_model = SVC().fit(X, y)
        prediction = svm_model.predict(input_df)

    elif model_option == 'Random Forest':
        rf_model = RandomForestClassifier(random_state=42).fit(X, y)
        prediction = rf_model.predict(input_df)

    elif model_option == 'Decision Tree':
        dt_model = DecisionTreeClassifier(random_state=42).fit(X, y)
        prediction = dt_model.predict(input_df)

    elif model_option == 'Gradient Boosting':
        gb_model = GradientBoostingClassifier(random_state=42).fit(X, y)
        prediction = gb_model.predict(input_df)

    # Display the prediction result
    st.write(f"Predicted Class: {prediction[0]}")
