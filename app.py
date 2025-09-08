import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import imblearn
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# App title
st.title("Diabetes Prediction App")
st.markdown("""
This application predicts whether a person is likely to have diabetes using a machine learning model trained on the Pima Indians Diabetes dataset.
""")

# Load data
@st.cache_data
def load_data():
    # Dynamically find the file path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "diabetes.csv")

    st.write(f"Loading dataset from: {file_path}")  # Debugging print

    df = pd.read_csv(file_path)
    return df

# EDA Options
if st.checkbox("Show Raw Dataset"):
    st.subheader("Raw Dataset")
    st.dataframe(df)

if st.checkbox("Show Class Distribution"):
    st.subheader("Class Distribution")
    st.bar_chart(df["Outcome"].value_counts())

# User input
st.subheader("Enter Patient Data for Prediction")
features = df.drop("Outcome", axis=1).columns

user_input = {}
for feature in features:
    user_input[feature] = st.slider(
        f"{feature}",
        float(df[feature].min()),
        float(df[feature].max()),
        float(df[feature].mean())
    )

input_df = pd.DataFrame([user_input])

# Data prep & SMOTE
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][prediction]

    st.subheader("Prediction Result")
    st.write("Diabetes Detected" if prediction == 1 else "No Diabetes Detected")
    st.write(f"Prediction Confidence: {pred_proba:.2f}")

# Correlation heatmap
st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Glucose histogram
st.subheader("Glucose Level Distribution")
fig, ax = plt.subplots()
sns.histplot(df["Glucose"], kde=True, ax=ax)
st.pyplot(fig)

# Pie chart
st.subheader("Diabetes Distribution in Dataset")
labels = ["No Diabetes", "Diabetes"]
sizes = df["Outcome"].value_counts()
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis("equal")
st.pyplot(fig)

# Evaluation
if st.checkbox("Show Model Evaluation"):
    y_pred = model.predict(X_test)

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Model Accuracy")
    st.write(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

    # Feature importance
    st.subheader("Feature Importances")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    st.bar_chart(importances.sort_values(ascending=True))

# Footer
st.markdown("---")
st.markdown("Made with Streamlit | [GitHub](https://github.com/Bioinformatic-geek/diabetes-app)")

