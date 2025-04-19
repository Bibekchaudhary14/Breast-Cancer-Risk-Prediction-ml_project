import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import sklearn
print(sklearn.__version__)

# Load the trained Random Forest model
model = load('random_forest_model.joblib')

# Create a Streamlit app
st.title("Breast Cancer Risk Predictions")

# Input fields for feature values on the main screen
st.header("Enter Patient Different Status")
PR_Status = st.selectbox("PR Status", ('Positive','Negative'))
ER_Status = st.selectbox("ER Status", ('Positive', 'Negative'))
Hormone_Therapy = st.selectbox("Hormone Therapy", ('Yes', 'No'))

Age_at_Diagnosis = st.number_input("Age at Diagnosis", min_value=10, max_value=90, value=25)
Tumor_Stage = st.number_input("Tumor Stage", min_value=0, max_value=4, value=2)
Tumor_Size = st.number_input("Tumor Size", min_value=10, max_value=80, value=20)

# Map input values to numeric using the label mapping
label_mapping_pr = {'Positive': 0,'Negative': 1}
PR_Status = label_mapping_pr[PR_Status]

label_mapping_er = {'Positive': 0,'Negative': 1}
ER_Status = label_mapping_er[ER_Status]

label_mapping_ht = {'Yes': 0,'No': 1}
Hormone_Therapy = label_mapping_ht[Hormone_Therapy]

# Make a prediction using the model
prediction = model.predict([[PR_Status, ER_Status, Hormone_Therapy, Age_at_Diagnosis, Tumor_Stage,Tumor_Size]])

# Display the prediction result on the main screen
st.header("Survival Prediction")
if prediction[0] == 0:
    st.error("Patient may be Deceased")
else:
    st.error("Patient is Living")