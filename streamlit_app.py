import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# Model evaluation
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

st.set_page_config(page_title="BITS ML Assignment 2: E-Commerce Churn Inference  App", layout="wide")
st.write("Upload a test dataset to run predictions across pre trained ML models.")

#1. Load pre trained models and preprocessor
@st.cache_resource
def load_artifacts():
    return joblib.load("models/churn_models.joblib")

try:
    artifacts = load_artifacts()
    preprocessor = artifacts['preprocessor']
    models = artifacts['models']

    #2. Upload test dataset
    uploaded_file = st.sidebar.file_uploader("Upload test csv file", type=['csv'])

    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)


        # Check for target lable 'Churn' in uploaded test dataset
        if 'Churn' not in test_df.columns:
            st.error('Uploaded CSV must contain the column name Churn')
        else:
            if 'CustomerID' in test_df.columns:
                test_tf = test_df.drop(columns=['CustomerID'])
            
            x_test = test_df.drop(columns=['Churn'])
            y_test = test_df['Churn']

            x_test_processed = preprocessor.transform(x_test)

            #3. Select model for evaluation
            st.subheader("Model evaluation")
            selected_model_name = st.selectbox("Choose a pre trained model", list(models.keys()))
            selected_model = models[selected_model_name]

            #4. Run inference and compute metrics 
            y_pred = selected_model.predict(x_test_processed)
            y_prob = selected_model.predict_proba(x_test_processed)[:, 1]

            # Calculate performance metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            # Metric display cards
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc * 100:.2f}%")
            col2.metric("Precision", f"{prec * 100:.2f}%")
            col3.metric("Recall", f"{rec * 100:.2f}%")
            col4.metric("F1-Score", f"{f1 * 100:.2f}%")
            col5.metric("MCC", f"{mcc:.4f}")
            col6.metric("ROC-AUC", f"{roc_auc * 100:.2f}%" if y_prob is not None else "N/A")

            st.divider()

            # 5. Display confusion matrix and classification report
            col_cm, col_report = st.columns(2)

            with col_cm:
                st.markdown("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                            xticklabels=['Retained (0)', 'Churned (1)'],
                            yticklabels=['Retained (0)', 'Churned (1)'])
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                st.pyplot(fig)

            with col_report:
                st.markdown("### Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
    else:
        st.info('Please upload a test dataset csv file in the sidebar')
except FileNotFoundError:
    st.error('Missing churn_mode.joblib')