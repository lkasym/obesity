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

def encode_input(input_data):
    return {
        'Age': (input_data['Age'] - dataset['Age'].mean()) / dataset['Age'].std(),
        'BMI': (input_data['BMI'] - dataset['BMI'].mean()) / dataset['BMI'].std(),
        'family_history_with_overweight': 1 if input_data['family_history_with_overweight'] == "Yes" else 0,
        'FAVC': 1 if input_data['FAVC'] == "Yes" else 0,
        'FCVC': ["Never", "Sometimes", "Always"].index(input_data['FCVC']),
        'NCP': float(["1 meal", "2 meals", "3 meals", "4 meals"].index(input_data['NCP']) + 1),
        'CAEC': ["No", "Sometimes", "Frequently", "Always"].index(input_data['CAEC']),
        'SMOKE': 1 if input_data['SMOKE'] == "Yes" else 0,
        'CH2O': ["Less than a liter", "Between 1 and 2 liters", "More than 2 liters"].index(input_data['CH2O']),
        'SCC': 1 if input_data['SCC'] == "Yes" else 0,
        'FAF': ["None", "1-2 days a week", "3-4 days a week", "4-5 days a week"].index(input_data['FAF']),
        'TUE': ["0-2 hours", "3-5 hours", "More than 5 hours"].index(input_data['TUE']),
        'CALC': ["No", "Sometimes", "Frequently", "Always"].index(input_data['CALC'])
    }

# Collecting user inputs
input_data = {
    'Age': st.number_input("Age", min_value=10, max_value=100),
    'BMI': st.number_input("Weight (in kgs)", min_value=30, max_value=200) / (st.number_input("Height (in meters)", min_value=1.0, max_value=2.5) ** 2),
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
    'CALC': st.selectbox("Frequency of alcohol consumption", ["No", "Sometimes", "Frequently", "Always"])
}

# Encode and standardize the input data
encoded_input = encode_input(input_data)
input_df = pd.DataFrame([encoded_input])

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
