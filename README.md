 Diabetes Prediction Web App

A simple web application built with Streamlit and scikit-learn that predicts whether a person is diabetic based on several medical parameters. The model uses the **PIMA Indian Diabetes Dataset** and incorporates **SMOTE** to balance the dataset.

 Features

- Predicts diabetes based on user input
- Interactive web interface using Streamlit
- Handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
- Clean and beginner-friendly code structure

---

 Technologies Used

- Python 
- pandas, numpy
- scikit-learn (ML models)
- imbalanced-learn (SMOTE)
- Streamlit (frontend)
- matplotlib / seaborn (for optional visualization)

---

  Dataset

This app uses the [PIMA Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), which includes the following features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (0 = No diabetes, 1 = Diabetes)

How to Run

1. Clone the repo
   ```bash
   git clone https://github.com/your-username/diabetes-prediction-app.git
   cd diabetes-prediction-app

