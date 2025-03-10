import numpy as np
import pandas as pd
df=pd.read_csv('healthcare.csv')
# finding Outliers

ul_bmi=df['bmi'].quantile(0.99)
ll_bmi=df['bmi'].quantile(0.01)

ul_age=df['age'].quantile(0.99)
ll_age=df['age'].quantile(0.01)

ul_g=df['avg_glucose_level'].quantile(0.99)
ll_g=df['avg_glucose_level'].quantile(0.01)

df_outliers=df[(df['bmi']>ul_bmi)|(df['bmi']<ll_bmi)|(df['age']>ul_age)|(df['age']<ll_age)|(df['avg_glucose_level']>ul_g)|(df['avg_glucose_level']<ll_g)]

# capping outliers
df['bmi']=np.where(df['bmi']>ul_bmi,ul_bmi,np.where(df['bmi']<ll_bmi,ll_bmi,df['bmi']))
df['age']=np.where(df['age']>ul_age,ul_age,np.where(df['age']<ll_age,ll_age,df['age']))
df['avg_glucose_level']=np.where(df['avg_glucose_level']>ul_g,ul_g,np.where(df['avg_glucose_level']<ll_g,ll_g,df['avg_glucose_level']))


from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


X=df[['gender','age','hypertension','ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status',]]
y=df[['heart_disease','stroke']]


ct=ColumnTransformer([
    ('imputer',SimpleImputer(strategy='mean'),['bmi']),
    ('OHE',OneHotEncoder(drop='first',sparse_output=False),['gender','ever_married',
       'work_type', 'Residence_type','smoking_status'])
],remainder='passthrough')

X_tf=ct.fit_transform(X)


# Train model
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=18, 
    min_samples_split=3, 
    min_samples_leaf=10, 
    class_weight="balanced",  # ⚠️ Fix for imbalanced data
    random_state=42
)
model.fit(X_tf, y)


import pickle
# Save the trained model
with open("final_rf_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save the preprocessor as well
with open("ct.pkl", "wb") as file:
    pickle.dump(ct, file)

print("Model & Preprocessor Saved ✅")





import streamlit as st
import pandas as pd
import pickle

# Load trained model & preprocessor
with open("final_rf_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("ct.pkl", "rb") as file:
    ct = pickle.load(file)

st.title("🩺 Early Disease Prediction System")

# User Input Fields
age = st.number_input("Enter Age:", min_value=1, max_value=100, value=30)
bmi = st.number_input("Enter BMI:", min_value=10.0, max_value=50.0, value=22.5, step=0.1)
glucose = st.number_input("Enter Glucose Level (min=50, max=300):", min_value=50, max_value=300, value=100)
gender = st.selectbox("Gender:", ["Male", "Female", "Other"])
smoking_status = st.selectbox("Smoking Status:", ["never smoked", "formerly smoked", "smokes", "Unknown"])
hypertension = st.selectbox("Do you suffer from hypertension?", [0, 1])  # Already 0/1, no need to map
ever_married = st.selectbox("Have you ever gotten married?", ["Yes", "No"])
work_type = st.selectbox("What is your profession?", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
Residence_type = st.selectbox("What is your residence type?", ["Urban", "Rural"])

# Convert input to DataFrame
user_input = pd.DataFrame([[gender, age, hypertension, ever_married, work_type, Residence_type, glucose, bmi, smoking_status]], 
                          columns=['gender', 'age', 'hypertension', 'ever_married',
                                   'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])

# Apply preprocessing
user_transformed = ct.transform(user_input)

# Predict button
if st.button("Predict Disease Risk"):
    prediction = model.predict(user_transformed)  # Get model prediction
    
    # Display results for all diseases
    for i, disease in enumerate(['Heart Disease', 'Stroke']):
        if prediction[0][i] == 1:
            st.error(f"⚠️ High Risk of {disease}! Consult a doctor.")
        else:
            st.success(f"✅ Low Risk of {disease} - Stay Healthy!")
