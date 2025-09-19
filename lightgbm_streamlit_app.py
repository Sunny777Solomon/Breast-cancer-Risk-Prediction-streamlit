import streamlit as st
import pandas as pd
import joblib

# Load the trained LR model
model = joblib.load('lr_model.pkl')

# Define all features used in training
feature_cols = [
    'Chemotherapy', 'ER status measured by IHC', 'ER Status', 'HER2 status measured by SNP6', 'HER2 Status',
    'Hormone Therapy', 'Inferred Menopausal State', 'Radio Therapy', 'PR Status', 'Tumor Stage_1.0',
    'Tumor Stage_2.0', 'Tumor Stage_3.0', 'Tumor Stage_4.0', 'Cancer Type_Breast Sarcoma',
    'Type of Breast Surgery_Mastectomy', 'Cellularity_Low', 'Cellularity_Moderate',
    'Pam50 + Claudin-low subtype_Her2', 'Pam50 + Claudin-low subtype_LumA',
    'Pam50 + Claudin-low subtype_LumB', 'Pam50 + Claudin-low subtype_NC',
    'Pam50 + Claudin-low subtype_Normal', 'Pam50 + Claudin-low subtype_claudin-low',
    'Cancer Type Detailed_Breast Angiosarcoma', 'Cancer Type Detailed_Breast Invasive Ductal Carcinoma',
    'Cancer Type Detailed_Breast Invasive Lobular Carcinoma', 'Cancer Type Detailed_Breast Invasive Mixed Mucinous Carcinoma',
    'Cancer Type Detailed_Breast Mixed Ductal and Lobular Carcinoma', 'Cancer Type Detailed_Invasive Breast Carcinoma',
    'Cancer Type Detailed_Metaplastic Breast Cancer', 'Neoplasm Histologic Grade_2.0',
    'Neoplasm Histologic Grade_3.0', '3-Gene classifier subtype_ER+/HER2- Low Prolif',
    '3-Gene classifier subtype_ER-/HER2-', '3-Gene classifier subtype_HER2+',
    'Tumor Other Histologic Subtype_Lobular', 'Tumor Other Histologic Subtype_Medullary',
    'Tumor Other Histologic Subtype_Metaplastic', 'Tumor Other Histologic Subtype_Mixed',
    'Tumor Other Histologic Subtype_Mucinous', 'Tumor Other Histologic Subtype_Other',
    'Tumor Other Histologic Subtype_Tubular/ cribriform', 'Primary Tumor Laterality_Right',
    'Integrative Cluster_10', 'Integrative Cluster_2', 'Integrative Cluster_3', 'Integrative Cluster_4ER+',
    'Integrative Cluster_4ER-', 'Integrative Cluster_5', 'Integrative Cluster_6', 'Integrative Cluster_7',
    'Integrative Cluster_8', 'Integrative Cluster_9', 'Age at Diagnosis', 'Lymph nodes examined positive',
    'Mutation Count', 'Nottingham prognostic index', 'Tumor Size'
]

st.title("10-Year Mortality Prediction App")

st.write("Fill in patient details below to predict 10-year mortality risk.")

# Create input widgets for some key features
age = st.slider('Age at Diagnosis', 20, 100, 50)
lymph_nodes = st.slider('Lymph nodes examined positive', 0, 50, 1)
mutation_count = st.slider('Mutation Count', 0, 500, 10)
npi = st.slider('Nottingham Prognostic Index', 0.0, 10.0, 4.5)
tumor_size = st.slider('Tumor Size', 0, 100, 20)
chemo = st.selectbox('Chemotherapy', [0, 1])
er_status = st.selectbox('ER Status', [0, 1])
pr_status = st.selectbox('PR Status', [0, 1])

# Create a base input dictionary
input_data = {
    'Age at Diagnosis': age,
    'Lymph nodes examined positive': lymph_nodes,
    'Mutation Count': mutation_count,
    'Nottingham prognostic index': npi,
    'Tumor Size': tumor_size,
    'Chemotherapy': chemo,
    'ER Status': er_status,
    'PR Status': pr_status
}

# Fill the rest with 0
for col in feature_cols:
    if col not in input_data:
        input_data[col] = 0

# Convert to DataFrame in correct order
input_df = pd.DataFrame([input_data])[feature_cols]

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted 10-Year Mortality: {'Yes' if prediction == 1 else 'No'}")
