import streamlit as st
import pandas as pd
from joblib import load

# Page Configuration
st.set_page_config(page_title="Obesity Prediction", layout="wide", page_icon=":chart_with_upwards_trend:")

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #f5f5f5;
        font-family: 'Helvetica Neue', sans-serif;
    }

    .big-font {
        font-size:24px !important;
        color: #4a4a4a;
    }

    .small-font {
        font-size:18px !important;
        color: #7a7a7a;
    }

    h1 {
        color: #ff6347;
        font-weight: bold;
    }

    .stButton>button {
        color: #f5f5f5;
        background-color: #ff6347;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }

    .stTextInput>div>div>input {
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }

    .stNumberInput>div>div>input {
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    
    .stSelectbox>div>div>select {
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }

    .stMarkdown {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
    }

</style>
""", unsafe_allow_html=True)

st.markdown("# Obesity Prediction")
st.markdown("### Please provide the following information to predict obesity level:")

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

# Decoding dictionary
obesity_mapping = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II',
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
}


# Health information based on research
health_info = {
    'Insufficient_Weight': {
        'Risks': ['Nutritional deficiencies', 'Osteoporosis', 'Anemia'],
        'Remedies': ['Increase calorie intake with nutritious foods', 'Consume protein-rich foods', 'Consultation with a nutritionist'],
        'Exercises': ['Strength training', 'Aerobic exercises'],
        'Citations': ["https://www.sciencedirect.com/science/article/pii/S1756464621000980"]
    },
    'Normal_Weight': {
        'Risks': ['Fewer health risks compared to other categories'],
        'Remedies': ['Maintain a balanced diet and regular exercise'],
        'Exercises': ['A mix of aerobic and strength training exercises'],
        'Citations': ["https://www.mdpi.com/1420-3049/21/10/1351"]
    },
    'Overweight_Level_I': {
        'Risks': ['Increased risk of metabolic syndrome', 'Cardiovascular diseases'],
        'Remedies': ['Green tea extract', 'Berberine', 'Ginger consumption'],
        'Exercises': ['High-intensity interval training', 'Strength training'],
        'Citations': [
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3727500",
            "https://www.sciencedirect.com/science/article/pii/S0753332220311768",
            "https://pubmed.ncbi.nlm.nih.gov/22538118/"
        ]
    },
    'Overweight_Level_II': {
        'Risks': ['Higher risk of metabolic syndrome', 'Type 2 diabetes', 'Cardiovascular diseases'],
        'Remedies': ['Cinnamon consumption', 'Turmeric and Curcumin'],
        'Exercises': ['Aerobic exercises', 'Resistance training'],
        'Citations': [
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3429799/",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3633300/"
        ]
    },
    # Assuming the information for 'Obesity_Type_I', 'Obesity_Type_II', and 'Obesity_Type_III' is similar to 'Overweight_Level_II'
    'Obesity_Type_I': {
        'Risks': ['Higher risk of metabolic syndrome', 'Type 2 diabetes', 'Cardiovascular diseases'],
        'Remedies': ['Cinnamon consumption', 'Turmeric and Curcumin'],
        'Exercises': ['Aerobic exercises', 'Resistance training'],
        'Citations': [
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3429799/",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3633300/"
        ]
    },
    'Obesity_Type_II': {
        'Risks': ['Higher risk of metabolic syndrome', 'Type 2 diabetes', 'Cardiovascular diseases'],
        'Remedies': ['Cinnamon consumption', 'Turmeric and Curcumin'],
        'Exercises': ['Aerobic exercises', 'Resistance training'],
        'Citations': [
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3429799/",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3633300/"
        ]
    },
    'Obesity_Type_III': {
        'Risks': ['Higher risk of metabolic syndrome', 'Type 2 diabetes', 'Cardiovascular diseases'],
        'Remedies': ['Cinnamon consumption', 'Turmeric and Curcumin'],
        'Exercises': ['Aerobic exercises', 'Resistance training'],
        'Citations': [
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3429799/",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3633300/"
        ]
    }
}



# Collect user input
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100)
    height = st.number_input("Height (in meters)", min_value=1.0, max_value=2.5)
with col2:
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
    decoded_prediction = obesity_mapping[prediction[0]]
    st.write(f"Predicted Class: {decoded_prediction}")

    # Display health info
    st.subheader("Health Information Based on Research")
    st.write(f"Potential Risks for {decoded_prediction}: {', '.join(health_info[decoded_prediction]['Risks'])}")
    st.write(f"Natural Remedies for {decoded_prediction}: {', '.join(health_info[decoded_prediction]['Remedies'])}")
    st.write(f"Recommended Exercises for {decoded_prediction}: {', '.join(health_info[decoded_prediction]['Exercises'])}")
    st.write("Research Citations:")
    for link in health_info[decoded_prediction]['Citations']:
        st.write(link)

# Disclaimer
st.write("The provided health risks, remedies, and exercises are based on research papers and may not be exhaustive. It's always recommended to consult with healthcare professionals for personalized advice.")
