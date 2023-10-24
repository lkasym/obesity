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

# Calculate mean and standard deviation for standardization
age_mean, age_std = dataset['Age'].mean(), dataset['Age'].std()
height_mean, height_std = dataset['Height'].mean(), dataset['Height'].std()
weight_mean, weight_std = dataset['Weight'].mean(), dataset['Weight'].std()

# Streamlit App
st.title("Obesity Prediction")

# Collecting user inputs
age = st.number_input("Age", min_value=10, max_value=100)
height = st.number_input("Height (in meters)", min_value=1.0, max_value=2.5)
weight = st.number_input("Weight (in kgs)", min_value=30, max_value=200)

# Encoding and Standardization
age = (age - age_mean) / age_std
height = (height - height_mean) / height_std
weight = (weight - weight_mean) / weight_std

family_history_with_overweight = 1 if st.selectbox("Family history with overweight?", ["Yes", "No"]) == "Yes" else 0
FAVC = 1 if st.selectbox("Frequent consumption of high caloric food?", ["Yes", "No"]) == "Yes" else 0
SMOKE = 1 if st.selectbox("Do you smoke?", ["Yes", "No"]) == "Yes" else 0
SCC = 1 if st.selectbox("Do you monitor calorie consumption?", ["Yes", "No"]) == "Yes" else 0

# Dropdown for Frequency of vegetable consumption
FCVC_options = ["Never", "Sometimes", "Always"]
FCVC = FCVC_options.index(st.selectbox("Frequency of vegetable consumption", FCVC_options))

# Dropdown for Number of main meals a day
NCP_options = ["1 meal", "2 meals", "3 meals", "4 meals"]
NCP = NCP_options.index(st.selectbox("Number of main meals a day", NCP_options)) + 1

# Dropdown for Consumption of food between meals
CAEC_options = ["No", "Sometimes", "Frequently", "Always"]
CAEC = CAEC_options.index(st.selectbox("Consumption of food between meals", CAEC_options))

# Dropdown for Daily water consumption
CH2O_options = ["Less than a liter", "Between 1 and 2 liters", "More than 2 liters"]
CH2O = CH2O_options.index(st.selectbox("Daily water consumption", CH2O_options)) + 1

# Dropdown for Frequency of physical activity
FAF_options = ["None", "1-2 days a week", "3-4 days a week", "4-5 days a week"]
FAF = FAF_options.index(st.selectbox("Frequency of physical activity", FAF_options))

# Dropdown for Time using technology devices daily
TUE_options = ["0-2 hours", "3-5 hours", "More than 5 hours"]
TUE = TUE_options.index(st.selectbox("Time using technology devices daily", TUE_options))

# Dropdown for Frequency of alcohol consumption
CALC_options = ["No", "Sometimes", "Frequently", "Always"]
CALC = CALC_options.index(st.selectbox("Frequency of alcohol consumption", CALC_options))

# Convert inputs to DataFrame for prediction
input_data = pd.DataFrame([[
    age, height, weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, 
    SMOKE, CH2O, SCC, FAF, TUE, CALC
]])

# Dropdown to select model for prediction
model_option = st.selectbox(
    'Choose a model for prediction:',
    ('KNN', 'Logistic Regression', 'Naive Bayes', 'SVM', 'Random Forest', 'Decision Tree', 'Gradient Boosting')
)

# Button to predict
if st.button("Predict"):
    # Load the provided dataset
    dataset = pd.read_csv(dataset_path)
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
