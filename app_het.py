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

# Load the dataset for feature preprocessing
dataset_path = "preprocessed_obesity_dataset_updated.csv"
dataset = pd.read_csv(dataset_path)

# Streamlit App
st.title("Obesity Prediction")

# Collecting user inputs
age = st.number_input("Age", min_value=10, max_value=100)
height = st.number_input("Height (in meters)", min_value=1.0, max_value=2.5)
weight = st.number_input("Weight (in kgs)", min_value=30, max_value=200)
bmi = weight / (height ** 2)

# Convert the categorical inputs to the format used in the training dataset
input_data = {
    'Age': age,
    'Height': height,
    'Weight': weight,
    'family_history_with_overweight': 1 if st.selectbox("Family history with overweight?", ["Yes", "No"]) == "Yes" else 0,
    'FAVC': 1 if st.selectbox("Frequent consumption of high caloric food?", ["Yes", "No"]) == "Yes" else 0,
    'FCVC': ['Never', 'Sometimes', 'Always'].index(st.selectbox("Frequency of vegetable consumption", ["Never", "Sometimes", "Always"])) + 1,
    'NCP': float(['1 meal', '2 meals', '3 meals', '4 meals'].index(st.selectbox("Number of main meals a day", ["1 meal", "2 meals", "3 meals", "4 meals"])) + 1),
    'CAEC': ['No', 'Sometimes', 'Frequently', 'Always'].index(st.selectbox("Consumption of food between meals", ["No", "Sometimes", "Frequently", "Always"])),
    'SMOKE': 1 if st.selectbox("Do you smoke?", ["Yes", "No"]) == "Yes" else 0,
    'CH2O': ['Less than a liter', 'Between 1 and 2 liters', 'More than 2 liters'].index(st.selectbox("Daily water consumption", ["Less than a liter", "Between 1 and 2 liters", "More than 2 liters"])) + 1,
    'SCC': 1 if st.selectbox("Do you monitor calorie consumption?", ["Yes", "No"]) == "Yes" else 0,
    'FAF': ['None', '1-2 days a week', '3-4 days a week', '4-5 days a week'].index(st.selectbox("Frequency of physical activity", ["None", "1-2 days a week", "3-4 days a week", "4-5 days a week"])),
    'TUE': ['0-2 hours', '3-5 hours', 'More than 5 hours'].index(st.selectbox("Time using technology devices daily", ["0-2 hours", "3-5 hours", "More than 5 hours"])),
    'CALC': ['No', 'Sometimes', 'Frequently', 'Always'].index(st.selectbox("Frequency of alcohol consumption", ["No", "Sometimes", "Frequently", "Always"]))
}

input_df = pd.DataFrame([input_data])

# Ensure that the input data columns match the training data columns
input_df = input_df[dataset.drop('NObeyesdad', axis=1).columns]

# Dropdown to select model for prediction
model_option = st.selectbox(
    'Choose a model for prediction:',
    ('KNN', 'Logistic Regression', 'Naive Bayes', 'SVM', 'Random Forest', 'Decision Tree', 'Gradient Boosting')
)

# Button to predict
if st.button("Predict"):
    # Split the dataset
    X = dataset.drop('NObeyesdad', axis=1)
    y = dataset['NObeyesdad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection and prediction
    if model_option == 'KNN':
        model = KNeighborsClassifier().fit(X_train, y_train)
    elif model_option == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    elif model_option == 'Naive Bayes':
        model = GaussianNB().fit(X_train, y_train)
    elif model_option == 'SVM':
        model = SVC().fit(X_train, y_train)
    elif model_option == 'Random Forest':
        model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    elif model_option == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    elif model_option == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)

    prediction = model.predict(input_df)

    # Display prediction
    st.write(f"Predicted Class: {prediction[0]}")
