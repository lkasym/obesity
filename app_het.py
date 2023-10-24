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

# Load the dataset
dataset_path = "preprocessed_obesity_dataset_updated.csv"
dataset = pd.read_csv(dataset_path)

# Streamlit App
st.title("Obesity Prediction")

# Collecting user inputs
age = st.number_input("Age", min_value=10, max_value=100)
height = st.number_input("Height (in meters)", min_value=1.0, max_value=2.5)
weight = st.number_input("Weight (in kgs)", min_value=30, max_value=200)
bmi = weight / (height ** 2)

# Encoding categorical variables
family_history_with_overweight = st.selectbox("Family history with overweight?", ["yes", "no"])
FAVC = st.selectbox("Frequent consumption of high caloric food?", ["yes", "no"])
FCVC = st.selectbox("Frequency of vegetable consumption", ["Never", "Sometimes", "Always"])
NCP = st.selectbox("Number of main meals a day", ["1 meal", "2 meals", "3 meals", "4 meals"])
CAEC = st.selectbox("Consumption of food between meals", ["No", "Sometimes", "Frequently", "Always"])
SMOKE = st.selectbox("Do you smoke?", ["yes", "no"])
CH2O = st.selectbox("Daily water consumption", ["Less than a liter", "Between 1 and 2 liters", "More than 2 liters"])
SCC = st.selectbox("Do you monitor calorie consumption?", ["yes", "no"])
FAF = st.selectbox("Frequency of physical activity", ["None", "1-2 days a week", "3-4 days a week", "4-5 days a week"])
TUE = st.selectbox("Time using technology devices daily", ["0-2 hours", "3-5 hours", "More than 5 hours"])
CALC = st.selectbox("Frequency of alcohol consumption", ["No", "Sometimes", "Frequently", "Always"])

# Convert inputs to DataFrame for prediction
input_data = pd.DataFrame([[
    age, height, weight, bmi, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC
]])

# Dropdown to select model for prediction
model_option = st.selectbox(
    'Choose a model for prediction:',
    ('KNN', 'Logistic Regression', 'Naive Bayes', 'SVM', 'Random Forest', 'Decision Tree', 'Gradient Boosting')
)

# Button to predict
if st.button("Predict"):
    # Prepare the dataset for training and testing
    X = dataset.drop('NObeyesdad', axis=1)
    y = dataset['NObeyesdad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predict using the chosen model
    prediction = None
    if model_option == 'KNN':
        knn_model = KNeighborsClassifier().fit(X_train, y_train)
        prediction = knn_model.predict(input_data)
    elif model_option == 'Logistic Regression':
        lr_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        prediction = lr_model.predict(input_data)
    elif model_option == 'Naive Bayes':
        nb_model = GaussianNB().fit(X_train, y_train)
        prediction = nb_model.predict(input_data)
    elif model_option == 'SVM':
        svm_model = SVC().fit(X_train, y_train)
        prediction = svm_model.predict(input_data)
    elif model_option == 'Random Forest':
        rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
        prediction = rf_model.predict(input_data)
    elif model_option == 'Decision Tree':
        dt_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
        prediction = dt_model.predict(input_data)
    elif model_option == 'Gradient Boosting':
        gb_model = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
        prediction = gb_model.predict(input_data)

    # Display the prediction result
    st.write(f"Predicted Class: {prediction[0]}")
