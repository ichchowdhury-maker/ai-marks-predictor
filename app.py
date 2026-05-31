import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("students.csv")

X = df[["Hours", "Sleep"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

st.title("AI Marks Predictor")

hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0)
sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0)

if st.button("Predict Marks"):
    prediction = model.predict([[hours, sleep]])
    st.success(f"Predicted Marks: {prediction[0]:.2f}")