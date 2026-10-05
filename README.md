# ❤️ Heart Disease Prediction using Machine Learning

An end-to-end **Explainable Machine Learning application** for predicting heart disease risk from patient health parameters.

The project compares multiple machine learning algorithms, performs leakage-free preprocessing, tunes a Random Forest model using GridSearchCV, validates the model using cross-validation, and deploys the final model through an interactive Streamlit application with **SHAP-based Explainable AI**.

---

## 🚀 Live Application

The Streamlit deployment link will be added here after deployment.

**Live Demo:** Coming soon

---

## 📌 Project Overview

The objective of this project is to build a machine learning system that can estimate the probability of heart disease based on clinical and demographic patient parameters.

Instead of simply returning a binary prediction, the upgraded application provides:

* ❤️ Heart disease prediction
* 📊 Disease probability score
* 🟢🟡🔴 Risk classification
* 📋 Patient health dashboard
* 🤖 Comparison of multiple ML algorithms
* 🔍 SHAP-based Explainable AI
* 📄 Downloadable patient prediction report
* 🌐 Interactive Streamlit web application

---

## 🧠 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Train / Test Split
   ↓
Leakage-Free Preprocessing
   ↓
Multiple ML Models
   ↓
Model Comparison
   ↓
Random Forest Selection
   ↓
Hyperparameter Tuning
   ↓
Cross-Validation
   ↓
Final Tuned Random Forest
   ↓
SHAP Explainability
   ↓
Streamlit Application
   ↓
Downloadable Patient Report
```

---

## 📊 Dataset

The project uses a heart disease dataset containing:

* **920 patient records**
* **13 predictive input features**
* **1 target variable**

The dataset contains demographic, clinical, and diagnostic parameters such as:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Fasting blood sugar
* Resting ECG
* Maximum heart rate
* Exercise-induced angina
* Oldpeak
* Slope
* Number of major vessels
* Thal

---

## 🤖 Models Compared

The following machine learning algorithms were evaluated:

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   84.24% |    84.11% | 88.24% |   86.12% |
| Decision Tree       |   79.35% |    80.19% | 83.33% |   81.73% |
| Random Forest       |   86.96% |    84.82% | 93.14% |   88.79% |
| SVM                 |   85.33% |    83.19% | 92.16% |   87.44% |
| KNN                 |   85.33% |    85.05% | 89.22% |   87.08% |
| Gradient Boosting   |   82.61% |    83.02% | 86.27% |   84.62% |

Random Forest demonstrated the strongest overall performance among the baseline models.

---

## 🏆 Final Tuned Random Forest

The Random Forest model was further optimized using **GridSearchCV**.

### Best Hyperparameters

```text
n_estimators      = 200
max_depth         = 10
min_samples_split = 5
min_samples_leaf  = 2
```

### Tuned Cross-Validation Performance

| Metric    | Mean Score |
| --------- | ---------: |
| Accuracy  |     83.15% |
| Precision |     82.42% |
| Recall    |     88.42% |
| F1 Score  |     85.27% |
| ROC-AUC   |     89.33% |

---

## 📈 Final Test Performance

The tuned model was evaluated on a completely held-out test set containing **184 samples**.

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **94.02%** |
| Precision | **92.52%** |
| Recall    | **97.06%** |
| F1 Score  | **94.74%** |
| ROC-AUC   | **98.89%** |

### Confusion Matrix

```text
[[74   8]
 [ 3  99]]
```

The model correctly identified **99 of 102 positive cases** and **74 of 82 negative cases** in the held-out test set.

---

## 🔍 Explainable AI — SHAP

The application uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions made by the Random Forest model.

SHAP helps answer:

> "Why did the model make this prediction for this particular patient?"

The application displays the features that contributed most strongly toward or away from the predicted heart disease class.

Important features identified by the model include:

* Exercise-induced angina
* Oldpeak
* Chest pain type
* Cholesterol
* Maximum heart rate
* Sex
* Age
* Number of major vessels
* Thal
* Resting blood pressure

SHAP explanations are generated for the individual patient entered into the application.

---

## 🖥️ Streamlit Application

The application provides an interactive interface where users can enter patient information and receive:

### 1. Prediction

```text
High Risk of Heart Disease
```

or

```text
Low Risk of Heart Disease
```

### 2. Probability

The application displays the model's estimated probability for:

* No heart disease
* Heart disease

### 3. Risk Band

The application categorizes the model probability into:

```text
🟢 Low Risk
🟡 Moderate Risk
🔴 High Risk
```

These categories are application-level interpretations and are **not clinically validated diagnostic thresholds**.

### 4. Patient Dashboard

The entered patient information is displayed in a structured health dashboard.

### 5. Model Comparison

The application displays the performance of the evaluated machine learning algorithms.

### 6. SHAP Explanation

The application displays the major factors influencing the individual prediction.

### 7. Downloadable Patient Report

Users can download a text-based report containing:

* Patient information
* Prediction
* Disease probability
* Risk band
* Patient observations
* Top SHAP contributions
* Final model information
* Model performance
* Disclaimer

---

## 🛠️ Technologies Used

### Programming

* Python

### Data Processing

* NumPy
* Pandas

### Machine Learning

* Scikit-learn
* Random Forest
* Logistic Regression
* Decision Tree
* SVM
* KNN
* Gradient Boosting
* GridSearchCV
* Cross-validation

### Explainable AI

* SHAP

### Deployment / Interface

* Streamlit

### Model Persistence

* Joblib

---

## 📁 Project Structure

```text
HeartAttack-Prediction/
│
├── app.py
│
├── heart_disease_final_pipeline.pkl
│
├── requirements.txt
│
├── README.md
│
├── dataset.csv
│
└── Heart_Attack_Prediction_Model.ipynb
```

### File Description

| File                                  | Purpose                            |
| ------------------------------------- | ---------------------------------- |
| `app.py`                              | Streamlit application              |
| `heart_disease_final_pipeline.pkl`    | Final tuned ML pipeline            |
| `requirements.txt`                    | Python dependencies                |
| `README.md`                           | Project documentation              |
| `dataset.csv`                         | Dataset used for model development |
| `Heart_Attack_Prediction_Model.ipynb` | Complete ML development workflow   |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/lokesh-ts/HeartAttack-Prediction.git
```

Navigate into the project:

```bash
cd HeartAttack-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📦 Requirements

The project uses:

```text
streamlit
pandas
numpy
scikit-learn==1.6.1
shap==0.52.0
joblib
```

Scikit-learn is pinned to version **1.6.1** to maintain compatibility with the saved model pipeline.

---

## 🔐 Machine Learning Pipeline

The final model uses a preprocessing pipeline to ensure that transformations are learned only from the training data.

This helps prevent **data leakage** during model development.

The final saved `.pkl` file contains the preprocessing and tuned Random Forest model required for inference.

---

## ⚠️ Important Disclaimer

This project is developed for **educational, research, and portfolio purposes**.

The predictions produced by this application are generated by a machine learning model and should **not be considered a medical diagnosis**.

The application has not been clinically validated and should not be used as a substitute for professional medical assessment.

---

## 👨‍💻 Author

**Lokesh**

B.Tech — Electronics and Communication Engineering

---

## ⭐ Project Highlights

* End-to-end machine learning workflow
* Leakage-free preprocessing
* Multiple model comparison
* Hyperparameter optimization
* Cross-validation
* Tuned Random Forest
* SHAP Explainable AI
* Individual prediction explanations
* Risk probability estimation
* Interactive Streamlit dashboard
* Downloadable patient reports
* Deployment-ready architecture

---

## 🔮 Future Improvements

Potential future enhancements include:

* Cloud deployment
* PDF patient report generation
* Model monitoring
* Improved calibration of predicted probabilities
* Larger and more diverse datasets
* External validation on independent datasets
* Clinical validation with healthcare professionals
