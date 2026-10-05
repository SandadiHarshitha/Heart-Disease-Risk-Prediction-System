import streamlit as st
import joblib
import pandas as pd
import shap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# ============================================================
# LOAD FINAL MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_final_pipeline.pkl")


final_model = load_model()

preprocessor = final_model.named_steps["preprocessor"]
rf_model = final_model.named_steps["classifier"]

explainer = shap.TreeExplainer(rf_model)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("❤️ Heart Disease Prediction using Machine Learning")

st.write(
    "This application predicts heart disease risk based on patient "
    "health parameters using a tuned Random Forest Machine Learning model."
)

st.info(
    "⚠️ This application is for educational and research purposes only "
    "and should not be used as a medical diagnosis."
)


# ============================================================
# PATIENT INPUTS
# ============================================================

st.subheader("👤 Enter Patient Details")


age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=50
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

cp = st.selectbox(
    "Chest Pain Type",
    [
        "typical angina",
        "atypical angina",
        "non-anginal",
        "asymptomatic"
    ]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50.0,
    max_value=250.0,
    value=120.0
)

chol = st.number_input(
    "Cholesterol Level",
    min_value=50.0,
    max_value=600.0,
    value=200.0
)

fbs = st.selectbox(
    "Fasting Blood Sugar",
    [False, True]
)

restecg = st.selectbox(
    "Rest ECG",
    [
        "normal",
        "lv hypertrophy",
        "st-t abnormality"
    ]
)

thalch = st.number_input(
    "Maximum Heart Rate",
    min_value=50.0,
    max_value=250.0,
    value=150.0
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [False, True]
)

oldpeak = st.number_input(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

slope = st.selectbox(
    "Slope",
    [
        "flat",
        "upsloping",
        "downsloping"
    ]
)

ca = st.number_input(
    "Number of Major Vessels",
    min_value=0,
    max_value=4,
    value=0
)

thal = st.selectbox(
    "Thal",
    [
        "normal",
        "fixed defect",
        "reversable defect"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("🔍 Predict Heart Disease Risk"):

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "age": [age],
        "trestbps": [trestbps],
        "chol": [chol],
        "thalch": [thalch],
        "oldpeak": [oldpeak],
        "ca": [ca],
        "sex": [sex],
        "cp": [cp],
        "fbs": [fbs],
        "restecg": [restecg],
        "exang": [exang],
        "slope": [slope],
        "thal": [thal]
    })


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = final_model.predict(input_data)[0]

    probability = final_model.predict_proba(input_data)[0]

    no_disease_probability = probability[0] * 100
    disease_probability = probability[1] * 100


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.subheader("🩺 Prediction Result")

    if prediction == 1:

        prediction_text = "High Risk of Heart Disease"

        st.error("⚠️ High Risk of Heart Disease")

    else:

        prediction_text = "Low Risk of Heart Disease"

        st.success("✅ Low Risk of Heart Disease")


    # ========================================================
    # RISK SCORE
    # ========================================================

    st.subheader("📊 Heart Disease Risk Score")

    risk_score = disease_probability


    if risk_score < 30:

        risk_band = "Low Risk"

        st.success("🟢 Low Risk")

    elif risk_score < 70:

        risk_band = "Moderate Risk"

        st.warning("🟡 Moderate Risk")

    else:

        risk_band = "High Risk"

        st.error("🔴 High Risk")


    st.metric(
        label="Model-Estimated Heart Disease Probability",
        value=f"{risk_score:.2f}%"
    )

    st.progress(
        int(round(risk_score))
    )

    st.caption(
        "Risk bands are application-level categories for model "
        "interpretation and are not clinically validated diagnostic thresholds."
    )


    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    st.write("### Prediction Probability")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "No Heart Disease",
            f"{no_disease_probability:.2f}%"
        )

    with col2:

        st.metric(
            "Heart Disease",
            f"{disease_probability:.2f}%"
        )


    # ========================================================
    # PATIENT HEALTH DASHBOARD
    # ========================================================

    st.subheader("📋 Patient Health Dashboard")

    dashboard_data = pd.DataFrame({

        "Parameter": [
            "Age",
            "Sex",
            "Chest Pain Type",
            "Resting Blood Pressure",
            "Cholesterol",
            "Fasting Blood Sugar",
            "Rest ECG",
            "Maximum Heart Rate",
            "Exercise Induced Angina",
            "Oldpeak",
            "Slope",
            "Major Vessels",
            "Thal"
        ],

        "Value": [
            f"{age} years",
            sex,
            cp,
            f"{trestbps:.0f} mmHg",
            f"{chol:.0f} mg/dL",
            str(fbs),
            restecg,
            f"{thalch:.0f} bpm",
            str(exang),
            f"{oldpeak:.2f}",
            slope,
            str(ca),
            thal
        ]
    })


    st.dataframe(
        dashboard_data,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PATIENT DATA OBSERVATIONS
    # ========================================================

    st.write("### 🔎 Patient Data Observations")

    observations = []


    if trestbps >= 140:

        observations.append(
            "⚠️ Resting blood pressure entered is relatively elevated."
        )

    elif trestbps < 90:

        observations.append(
            "ℹ️ Resting blood pressure entered is relatively low."
        )

    else:

        observations.append(
            "✅ Resting blood pressure is within the application's reference range."
        )


    if chol >= 240:

        observations.append(
            "⚠️ Cholesterol value entered is relatively high."
        )

    elif chol < 150:

        observations.append(
            "ℹ️ Cholesterol value entered is relatively low."
        )

    else:

        observations.append(
            "✅ Cholesterol value is within the application's reference range."
        )


    if thalch < 120:

        observations.append(
            "ℹ️ Maximum heart rate entered is relatively low."
        )

    elif thalch >= 180:

        observations.append(
            "ℹ️ Maximum heart rate entered is relatively high."
        )

    else:

        observations.append(
            "✅ Maximum heart rate is within the application's reference range."
        )


    if oldpeak >= 2:

        observations.append(
            "⚠️ Oldpeak value entered is relatively elevated."
        )

    else:

        observations.append(
            "✅ Oldpeak value is below the application's highlighted threshold."
        )


    if ca > 0:

        observations.append(
            "ℹ️ A non-zero number of major vessels was entered."
        )

    else:

        observations.append(
            "✅ No major vessel indicator was entered."
        )


    if exang:

        observations.append(
            "ℹ️ Exercise-induced angina was selected."
        )

    else:

        observations.append(
            "ℹ️ Exercise-induced angina was not selected."
        )


    for observation in observations:

        st.write(observation)


    st.caption(
        "These observations summarize entered values and model inputs. "
        "They are not medical diagnoses or clinical recommendations."
    )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.subheader("🤖 Machine Learning Model Comparison")

    comparison_data = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "SVM",
            "KNN",
            "Gradient Boosting"
        ],

        "Accuracy": [
            0.842391,
            0.793478,
            0.869565,
            0.853261,
            0.853261,
            0.826087
        ],

        "Precision": [
            0.841121,
            0.801887,
            0.848214,
            0.831858,
            0.850467,
            0.830189
        ],

        "Recall": [
            0.882353,
            0.833333,
            0.931373,
            0.921569,
            0.892157,
            0.862745
        ],

        "F1 Score": [
            0.861244,
            0.817308,
            0.887850,
            0.874419,
            0.870813,
            0.846154
        ]
    })


    comparison_display = comparison_data.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    for column in metric_columns:

        comparison_display[column] = (
            comparison_display[column] * 100
        ).round(2).astype(str) + "%"


    st.dataframe(
        comparison_display,
        use_container_width=True,
        hide_index=True
    )


    st.write("### 📈 Model Performance Comparison")

    chart_data = comparison_data.set_index("Model")[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ]

    st.bar_chart(chart_data)


    # ========================================================
    # FINAL TUNED RANDOM FOREST
    # ========================================================

    st.write("### 🏆 Final Selected Model: Tuned Random Forest")


    tuned_results = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],

        "Score": [
            "94.02%",
            "92.52%",
            "97.06%",
            "94.74%",
            "98.89%"
        ]
    })


    st.dataframe(
        tuned_results,
        use_container_width=True,
        hide_index=True
    )


    st.success(
        "The Random Forest model was selected for further hyperparameter "
        "tuning and achieved improved performance on the held-out test set."
    )


    # ========================================================
    # SHAP EXPLAINABLE AI
    # ========================================================

    st.subheader("🔍 Explainable AI")

    st.write(
        "SHAP (SHapley Additive exPlanations) shows which patient "
        "features influenced this prediction."
    )


    processed_input = preprocessor.transform(input_data)

    patient_shap = explainer.shap_values(processed_input)

    # SHAP 0.52.0:
    # Shape = (samples, features, classes)

    patient_shap_disease = patient_shap[:, :, 1][0]

    feature_names = preprocessor.get_feature_names_out()


    shap_df = pd.DataFrame({

        "Feature": feature_names,

        "SHAP Value": patient_shap_disease
    })


    shap_df["Absolute SHAP"] = shap_df["SHAP Value"].abs()


    shap_df = shap_df.sort_values(
        "Absolute SHAP",
        ascending=False
    )


    # --------------------------------------------------------
    # Human-readable feature names
    # --------------------------------------------------------

    feature_name_mapping = {

        "num__age":
            "Age",

        "num__trestbps":
            "Resting Blood Pressure",

        "num__chol":
            "Cholesterol",

        "num__thalch":
            "Maximum Heart Rate",

        "num__oldpeak":
            "Oldpeak",

        "num__ca":
            "Number of Major Vessels",

        "cat__sex_Male":
            "Sex: Male",

        "cat__cp_atypical angina":
            "Chest Pain: Atypical Angina",

        "cat__cp_non-anginal":
            "Chest Pain: Non-Anginal",

        "cat__cp_typical angina":
            "Chest Pain: Typical Angina",

        "cat__fbs_True":
            "Fasting Blood Sugar: True",

        "cat__restecg_normal":
            "Rest ECG: Normal",

        "cat__restecg_st-t abnormality":
            "Rest ECG: ST-T Abnormality",

        "cat__exang_True":
            "Exercise-Induced Angina",

        "cat__slope_flat":
            "Slope: Flat",

        "cat__slope_upsloping":
            "Slope: Upsloping",

        "cat__thal_normal":
            "Thal: Normal",

        "cat__thal_reversable defect":
            "Thal: Reversible Defect"
    }


    shap_df["Feature"] = shap_df["Feature"].replace(
        feature_name_mapping
    )


    # ========================================================
    # TOP SHAP FEATURES
    # ========================================================

    st.write(
        "### 📊 Top Factors Influencing This Prediction"
    )


    top_features = shap_df.head(8)


    for _, row in top_features.iterrows():

        feature = row["Feature"]

        value = row["SHAP Value"]


        if value > 0:

            st.write(
                f"🔴 **{feature}** → contributed toward the "
                f"Heart Disease prediction ({value:+.4f})"
            )

        else:

            st.write(
                f"🔵 **{feature}** → contributed toward the "
                f"No Heart Disease prediction ({value:+.4f})"
            )


    # ========================================================
    # ALL SHAP VALUES
    # ========================================================

    with st.expander("View All SHAP Contributions"):

        display_df = shap_df[
            ["Feature", "SHAP Value"]
        ].copy()


        st.dataframe(
            display_df,
            use_container_width=True
        )


    # ========================================================
    # DOWNLOADABLE PATIENT REPORT
    # ========================================================

    st.subheader("📄 Patient Report")

    st.write(
        "Generate a downloadable text report containing the "
        "patient inputs, prediction, risk score, observations, "
        "and SHAP-based explanation."
    )


    # --------------------------------------------------------
    # Prepare SHAP report
    # --------------------------------------------------------

    shap_report = ""

    for _, row in top_features.iterrows():

        feature = row["Feature"]

        value = row["SHAP Value"]

        if value > 0:

            direction = "Toward Heart Disease"

        else:

            direction = "Toward No Heart Disease"


        shap_report += (
            f"- {feature}: "
            f"{value:+.4f} "
            f"({direction})\n"
        )


    # --------------------------------------------------------
    # Prepare observations report
    # --------------------------------------------------------

    observations_report = ""

    for observation in observations:

        # Remove emoji characters from report text
        clean_observation = (
            observation
            .replace("⚠️", "")
            .replace("ℹ️", "")
            .replace("✅", "")
            .strip()
        )

        observations_report += (
            f"- {clean_observation}\n"
        )


    # --------------------------------------------------------
    # Prediction label
    # --------------------------------------------------------

    if prediction == 1:

        report_prediction = "High Risk of Heart Disease"

    else:

        report_prediction = "Low Risk of Heart Disease"


    # --------------------------------------------------------
    # Complete report
    # --------------------------------------------------------

    report = f"""
============================================================
             HEART DISEASE PREDICTION REPORT
============================================================

Generated by:
Heart Disease Prediction using Machine Learning

------------------------------------------------------------
PATIENT DETAILS
------------------------------------------------------------

Age                         : {age} years
Sex                         : {sex}
Chest Pain Type             : {cp}
Resting Blood Pressure      : {trestbps:.0f} mmHg
Cholesterol                 : {chol:.0f} mg/dL
Fasting Blood Sugar         : {fbs}
Rest ECG                    : {restecg}
Maximum Heart Rate          : {thalch:.0f} bpm
Exercise Induced Angina     : {exang}
Oldpeak                     : {oldpeak:.2f}
Slope                       : {slope}
Number of Major Vessels     : {ca}
Thal                        : {thal}


------------------------------------------------------------
PREDICTION
------------------------------------------------------------

Prediction                  : {report_prediction}

Heart Disease Probability   : {disease_probability:.2f}%
No Heart Disease Probability: {no_disease_probability:.2f}%

Risk Band                   : {risk_band}


------------------------------------------------------------
PATIENT DATA OBSERVATIONS
------------------------------------------------------------

{observations_report}

------------------------------------------------------------
SHAP EXPLAINABLE AI
------------------------------------------------------------

Top factors influencing this prediction:

{shap_report}

SHAP values indicate the contribution of individual features
to the model prediction.

Positive values:
Contribution toward the Heart Disease prediction.

Negative values:
Contribution toward the No Heart Disease prediction.


------------------------------------------------------------
FINAL MODEL
------------------------------------------------------------

Model                       : Tuned Random Forest
Hyperparameter Tuning       : GridSearchCV

Number of Trees             : 200
Maximum Tree Depth          : 10
Minimum Samples Split       : 5
Minimum Samples Leaf        : 2

------------------------------------------------------------
MODEL PERFORMANCE
------------------------------------------------------------

Test Accuracy              : 94.02%
Test Precision             : 92.52%
Test Recall                : 97.06%
Test F1 Score              : 94.74%
Test ROC-AUC                : 98.89%

Tuned Cross-Validation
Accuracy                   : 83.15%
Recall                     : 88.42%
F1 Score                   : 85.27%
ROC-AUC                    : 89.33%


------------------------------------------------------------
DISCLAIMER
------------------------------------------------------------

This report is generated by a machine learning model and is
intended for educational and research purposes only.

The prediction and risk categories are not clinically validated
and should not be considered a medical diagnosis.

Please consult a qualified healthcare professional for actual
medical assessment and treatment decisions.

============================================================
                  END OF REPORT
============================================================
"""


    # --------------------------------------------------------
    # Download button
    # --------------------------------------------------------

    st.download_button(

        label="📥 Download Patient Report",

        data=report,

        file_name="heart_disease_patient_report.txt",

        mime="text/plain"
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with st.expander("ℹ️ Model Information"):

        st.write(
            """
            **Machine Learning Model:** Tuned Random Forest

            **Hyperparameter Tuning:** GridSearchCV

            **Number of Trees:** 200

            **Maximum Tree Depth:** 10

            **Minimum Samples per Split:** 5

            **Minimum Samples per Leaf:** 2

            **Explainable AI:** SHAP

            **Baseline Cross-Validation ROC-AUC:** 87.82%

            **Tuned Cross-Validation Accuracy:** 83.15%

            **Tuned Cross-Validation Recall:** 88.42%

            **Tuned Cross-Validation F1 Score:** 85.27%

            **Tuned Cross-Validation ROC-AUC:** 89.33%

            **Held-Out Test Accuracy:** 94.02%

            **Held-Out Test Recall:** 97.06%

            **Held-Out Test ROC-AUC:** 98.89%
            """
        )