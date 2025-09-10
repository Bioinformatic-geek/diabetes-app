import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# -----------------------------
# App Title
# -----------------------------
st.title("🧪 Diabetes Prediction App")
st.write("Upload a CSV file or an image of your diabetes report to predict diabetes.")

# -----------------------------
# File Uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your diabetes report (CSV or Image JPG/PNG)", 
    type=["csv", "jpg", "jpeg", "png"]
)

# -----------------------------
# Function to extract features from OCR text
# -----------------------------
def extract_features_from_text(text):
    # Using regex to find numeric values for common features
    features = {}
    patterns = {
        "Pregnancies": r"Pregnancies[:\s]+(\d+)",
        "Glucose": r"Glucose[:\s]+(\d+)",
        "BloodPressure": r"BloodPressure[:\s]+(\d+)",
        "SkinThickness": r"SkinThickness[:\s]+(\d+)",
        "Insulin": r"Insulin[:\s]+(\d+)",
        "BMI": r"BMI[:\s]+([\d.]+)",
        "DiabetesPedigreeFunction": r"DiabetesPedigreeFunction[:\s]+([\d.]+)",
        "Age": r"Age[:\s]+(\d+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            features[key] = float(match.group(1))
        else:
            features[key] = 0.0  # default if not found
    return pd.DataFrame([features])

# -----------------------------
# Load and process uploaded file
# -----------------------------
if uploaded_file is not None:
    
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.subheader("Raw Dataset")
        st.dataframe(df)
        
    else:  # Image file
        st.image(uploaded_file, caption="Uploaded Report", use_column_width=True)
        image = Image.open(uploaded_file)
        
        # OCR
        text = pytesseract.image_to_string(image)
        st.subheader("Extracted Text from Report")
        st.text_area("Text Output", text)
        
        # Extract features from OCR text
        df = extract_features_from_text(text)
        st.subheader("Extracted Features")
        st.dataframe(df)

    # -----------------------------
    # Train Model on Pima Dataset
    # -----------------------------
    # Load original Pima dataset for training
    pima_data = pd.read_csv("C:/Users/Varshini/Downloads/diabetes.csv")  # change path as needed
    X = pima_data.drop("Outcome", axis=1)
    y = pima_data["Outcome"]

    # SMOTE to handle imbalance
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Train RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # -----------------------------
    # Make Prediction
    # -----------------------------
    if st.button("🔮 Predict"):
        # If image uploaded, use extracted df; if CSV, use last row
        if uploaded_file.name.endswith(".csv"):
            input_df = df.tail(1)  # use last row of CSV for prediction
        else:
            input_df = df  # features extracted from image

        prediction = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0][prediction]

        st.subheader("🧾 Prediction Result")
        st.write("Diabetes Detected" if prediction == 1 else "No Diabetes Detected")
        st.write(f"🔢 Prediction Confidence: {pred_proba:.2f}")

    # -----------------------------
    # Optional: Show Model Evaluation
    # -----------------------------
    if st.checkbox("📈 Show Model Evaluation"):
        y_pred = model.predict(X_test)
        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Model Accuracy")
        st.write(f"{accuracy_score(y_test, y_pred)*100:.2f}%")
