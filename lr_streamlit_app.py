import streamlit as st
import pandas as pd
import joblib

# Load the trained Logistic Regression model
model = joblib.load("lr_model.pkl")

# All features used during training
feature_columns = [
    'Chemotherapy', 'ER status measured by IHC', 'ER Status', 'HER2 status measured by SNP6',
    'HER2 Status', 'Hormone Therapy', 'Inferred Menopausal State', 'Radio Therapy', 'PR Status',
    'Tumor Stage_1.0', 'Tumor Stage_2.0', 'Tumor Stage_3.0', 'Tumor Stage_4.0',
    'Cancer Type_Breast Sarcoma', 'Type of Breast Surgery_Mastectomy', 'Cellularity_Low',
    'Cellularity_Moderate', 'Pam50 + Claudin-low subtype_Her2', 'Pam50 + Claudin-low subtype_LumA',
    'Pam50 + Claudin-low subtype_LumB', 'Pam50 + Claudin-low subtype_NC',
    'Pam50 + Claudin-low subtype_Normal', 'Pam50 + Claudin-low subtype_claudin-low',
    'Cancer Type Detailed_Breast Angiosarcoma', 'Cancer Type Detailed_Breast Invasive Ductal Carcinoma',
    'Cancer Type Detailed_Breast Invasive Lobular Carcinoma',
    'Cancer Type Detailed_Breast Invasive Mixed Mucinous Carcinoma',
    'Cancer Type Detailed_Breast Mixed Ductal and Lobular Carcinoma',
    'Cancer Type Detailed_Invasive Breast Carcinoma', 'Cancer Type Detailed_Metaplastic Breast Cancer',
    'Neoplasm Histologic Grade_2.0', 'Neoplasm Histologic Grade_3.0',
    '3-Gene classifier subtype_ER+/HER2- Low Prolif', '3-Gene classifier subtype_ER-/HER2-',
    '3-Gene classifier subtype_HER2+', 'Tumor Other Histologic Subtype_Lobular',
    'Tumor Other Histologic Subtype_Medullary', 'Tumor Other Histologic Subtype_Metaplastic',
    'Tumor Other Histologic Subtype_Mixed', 'Tumor Other Histologic Subtype_Mucinous',
    'Tumor Other Histologic Subtype_Other', 'Tumor Other Histologic Subtype_Tubular/ cribriform',
    'Primary Tumor Laterality_Right', 'Integrative Cluster_10', 'Integrative Cluster_2',
    'Integrative Cluster_3', 'Integrative Cluster_4ER+', 'Integrative Cluster_4ER-',
    'Integrative Cluster_5', 'Integrative Cluster_6', 'Integrative Cluster_7',
    'Integrative Cluster_8', 'Integrative Cluster_9', 'Age at Diagnosis',
    'Lymph nodes examined positive', 'Mutation Count', 'Nottingham prognostic index', 'Tumor Size'
]

# Streamlit app
st.title("Breast Cancer 10-Year Mortality Prediction (Logistic Regression)")

st.write("This app predicts whether a breast cancer patient is likely to **survive beyond 10 years** after diagnosis.")

# Collect user inputs for important features
age = st.slider("Age at Diagnosis", 20, 100, 55)
tumor_size = st.slider("Tumor Size (mm)", 0, 200, 30)
lymph_nodes = st.slider("Positive Lymph Nodes", 0, 50, 2)
mutation_count = st.slider("Mutation Count", 0, 500, 10)
npi = st.slider("Nottingham Prognostic Index", 0.0, 10.0, 4.5)

chemo = st.selectbox("Chemotherapy", [0, 1])
hormone = st.selectbox("Hormone Therapy", [0, 1])
radio = st.selectbox("Radiotherapy", [0, 1])
er_status = st.selectbox("ER Status", [0, 1])
pr_status = st.selectbox("PR Status", [0, 1])
her2_status = st.selectbox("HER2 Status", [0, 1])

# Build input data dictionary
input_data = {
    "Age at Diagnosis": age,
    "Tumor Size": tumor_size,
    "Lymph nodes examined positive": lymph_nodes,
    "Mutation Count": mutation_count,
    "Nottingham prognostic index": npi,
    "Chemotherapy": chemo,
    "Hormone Therapy": hormone,
    "Radio Therapy": radio,
    "ER Status": er_status,
    "PR Status": pr_status,
    "HER2 Status": her2_status
}

# Fill missing features with 0
for col in feature_columns:
    if col not in input_data:
        input_data[col] = 0

# Convert to DataFrame in correct order
input_df = pd.DataFrame([input_data])[feature_columns]

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "❌ Died within 10 Years" if prediction == 1 else "✅ Survived 10+ Years"
    st.subheader(f"Prediction: {result}")
