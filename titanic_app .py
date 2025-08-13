# ===============================================
# 🚢 Titanic Survival Predictor - Interactive App
# Author: Omii's ML Lab
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    le = LabelEncoder()
    for col in ['Sex', 'Embarked']:
        df[col] = le.fit_transform(df[col])
    X = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1)
    y = df['Survived']
    return X, y, le

X, y, le = load_data()

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to see survival prediction.")

# User inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
fare = st.slider("Ticket Fare", 0.0, 520.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])

# Convert inputs
sex_enc = 1 if sex == "Male" else 0
embarked_enc = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}[embarked]

# Predict
input_data = np.array([[pclass, sex_enc, age, sibsp, parch, fare, embarked_enc]])
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# Display result
if st.button("Predict Survival"):
    if prediction == 1:
        st.success(f"✅ Survived! Probability: {probability*100:.2f}%")
    else:
        st.error(f"❌ Did not survive. Probability: {probability*100:.2f}%")
