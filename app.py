import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Pima Indians Diabetes Prediction App")

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    df = pd.read_csv(url, names=columns)
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    return df

df = load_data()
st.subheader("Data Preview")
st.write(df.head())

# Split and scale
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

st.subheader("Model Performance")
st.write(classification_report(y_test, y_pred, output_dict=False))
st.write("ROC-AUC Score:", round(roc_auc_score(y_test, y_proba), 3))

# Threshold slider
threshold = st.slider("Set Threshold for Positive Prediction", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
y_pred_thresh = (y_proba >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_thresh)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix at Threshold {:.2f}'.format(threshold))
st.pyplot(fig)

# Feature importances
importances = pd.Series(rf.feature_importances_, index=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
st.subheader("Feature Importances")
st.bar_chart(importances)

# Prediction section
st.subheader("Predict Diabetes for New Patient Data")
input_values = []
for feature in X.columns:
    val = st.number_input(f"{feature}", value=float(df[feature].mean()))
    input_values.append(val)

if st.button("Predict"):
    input_scaled = scaler.transform([input_values])
    prob = rf.predict_proba(input_scaled)[0][1]
    pred = int(prob >= threshold)
    st.write(f"Predicted Probability of Diabetes: {prob:.2f}")
    st.write("Prediction: **Diabetic**" if pred else "Prediction: **Non-Diabetic**")
