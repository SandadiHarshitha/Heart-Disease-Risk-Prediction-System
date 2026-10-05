# ❤️ Heart Attack Prediction using Machine Learning Algorithms


## 📌 Project Overview

Heart disease is one of the major health concerns worldwide. Early prediction of heart disease risk can help in taking preventive measures.

This project aims to build a Machine Learning based prediction system that learns patterns from patient health records and predicts the possibility of heart disease.

A user-friendly Streamlit web application is developed where users can enter patient medical details and receive a prediction along with confidence and explanation of important risk factors.


## 🎯 Project Objective

The main objectives of this project are:

- Analyze important heart disease risk factors
- Perform data preprocessing and feature engineering
- Train multiple Machine Learning algorithms
- Compare model performances using evaluation metrics
- Select the best performing model
- Deploy the final trained model as a web application
- Implement Explainable AI techniques for understanding predictions


## 📂 Dataset Description

Dataset Used:

Heart Disease Dataset (based on UCI Heart Disease Dataset)

Number of Records:

920 patient records

Number of Features:

13 original medical features


The dataset contains patient information such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol level
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate achieved
- Exercise induced angina
- Oldpeak
- Slope
- Number of major vessels
- Thal


## 🛠️ Technologies Used

Programming Language:

- Python


Machine Learning Libraries:

- Pandas
- NumPy
- Scikit-learn


Visualization:

- Matplotlib


Deployment:

- Streamlit


Model Saving:

- Pickle (.pkl)


## 🔄 Project Workflow

The complete workflow followed in this project:
Data Collection

    ↓

Data Preprocessing

    ↓

Exploratory Data Analysis

    ↓

Feature Encoding

    ↓

Feature Scaling

    ↓

Model Training

    ↓

Model Evaluation

    ↓

Hyperparameter Tuning

    ↓

Model Selection

    ↓

Deployment using Streamlit



## 🤖 Machine Learning Algorithms Used

The following algorithms were trained and compared:

1. Logistic Regression

2. Decision Tree Classifier

3. Random Forest Classifier

4. Support Vector Machine (SVM)

5. K-Nearest Neighbors (KNN)

6. Gradient Boosting Classifier


## 🏆 Final Model Selection

After comparing different Machine Learning models, Random Forest Classifier was selected as the final model.

Reason for selecting Random Forest:

- Better accuracy compared to other models
- High recall value
- Better F1-score
- Handles complex patterns efficiently


## 📊 Final Model Performance

Final Model:

Random Forest Classifier


Performance:

Accuracy:

86.95%


Precision:

84.82%


Recall:

93.13%


F1 Score:

88.78%


The high recall score is important because in healthcare prediction systems it helps reduce the chances of missing high-risk patients.


## 🔍 Explainable AI Implementation

Explainable AI techniques were implemented to understand how the model makes predictions.

The project includes:

- Random Forest Feature Importance analysis
- Identification of important risk factors
- Patient-specific risk factor explanation
- Prediction confidence percentage


Important features identified by the model:

- Cholesterol
- Maximum heart rate
- Age
- Oldpeak
- Exercise induced angina
- Resting blood pressure


## 🌐 Streamlit Web Application

A web interface was developed using Streamlit.

The application allows users to:

- Enter patient health details
- Predict heart disease risk
- View prediction confidence
- Understand important factors influencing the prediction


## 📁 Project Structure

```text
HeartAttack Prediction
│
├── dataset.csv
│
├── Heart_Attack_Prediction_Model.ipynb
│
├── app.py
│
├── heart_model.pkl
│
├── scaler.pkl
│
├── requirements.txt
│
└── README.md
```



## ▶️ How to Run the Project


Install required dependencies:


pip install -r requirements.txt



Run the Streamlit application:


streamlit run app.py



The application will open in the browser.


## 📦 Required Libraries

The required packages are mentioned in:


requirements.txt



## ✅ Conclusion

This project demonstrates a complete Machine Learning lifecycle including:

- Data preprocessing
- Model training
- Model evaluation
- Explainable AI
- Model deployment

The final system predicts heart disease risk using a trained Random Forest Machine Learning model and provides an interactive interface for users.


## 👨‍💻 Author

Machine Learning Internship Project

Heart Attack Prediction using Machine Learning Algorithms