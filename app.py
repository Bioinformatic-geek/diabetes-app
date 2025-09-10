import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

st.title("🧪 Diabetes Prediction App")
st.write("Predict diabetes using a Random Forest Classifier.")

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("Upload Pima Diabetes CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Dataset")
    st.dataframe(df)

    # -----------------------------
    # Feature input for prediction
    # -----------------------------
    features = df.drop("Outcome", axis=1).columns
    user_input = {}
    for feature in features:
        user_input[feature] = st.slider(
            f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean())
        )
    input_df = pd.DataFrame([user_input])

    # -----------------------------
    # Train Model with SMOTE
    # -----------------------------
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("🔮 Predict"):
        prediction = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0][prediction]

        st.subheader("🧾 Prediction Result:")
        st.write("Diabetes Detected" if prediction == 1 else "No Diabetes Detected")
        st.write(f"🔢 Prediction Confidence: {pred_proba:.2f}")

    # -----------------------------
    # Model evaluation
    # -----------------------------
    if st.checkbox("📈 Show Model Evaluation"):
        y_pred = model.predict(X_test)
        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Model Accuracy")
        st.write(f"{accuracy_score(y_test, y_pred)*100:.2f}%")
