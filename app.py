import streamlit as st
import pandas as pd
import numpy as np
import imblearn
import matplotlib.pyplot as plt
st.write("scikit-learn version:", sklearn.__version__)
st.write("imbalanced-learn version:", imblearn.__version__)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# App title
st.title("🧪 Diabetes Prediction App")
st.write("Predict diabetes using the Pima Indian dataset and a Random Forest Classifier.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

df = load_data()

# EDA Section
if st.checkbox("🔍 Show Raw Dataset"):
    st.subheader("Raw Pima Indian Diabetes Dataset")
    st.dataframe(df)

if st.checkbox("📊 Show Class Distribution"):
    st.subheader("Class Distribution")
    st.bar_chart(df["Outcome"].value_counts())

# Feature input for prediction
st.subheader("🧠 Enter Patient Data for Prediction")
features = df.drop("Outcome", axis=1).columns

user_input = {}
for feature in features:
    user_input[feature] = st.slider(
        f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean())
    )

input_df = pd.DataFrame([user_input])

# SMOTE + Train Model
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
if st.button("🔮 Predict"):
    prediction = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][prediction]
    
    st.subheader("🧾 Prediction Result:")
    st.write("Diabetes Detected" if prediction == 1 else "No Diabetes Detected")
    st.write(f"🔢 Prediction Confidence: {pred_proba:.2f}")

# Evaluation Section
if st.checkbox("📈 Show Model Evaluation"):
    y_pred = model.predict(X_test)

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Model Accuracy")
    st.write(f"{accuracy_score(y_test, y_pred)*100:.2f}%")

    # Feature importance
    st.subheader("📌 Feature Importances")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    st.bar_chart(importances.sort_values(ascending=True))

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | [GitHub](https://github.com/yourusername)")

