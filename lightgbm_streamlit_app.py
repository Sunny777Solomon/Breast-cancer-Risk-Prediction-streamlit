import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional medical theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-positive {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff4757;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .prediction-negative {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2ed573;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load the trained LR model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('lr_model.pkl')
        return model, None
    except FileNotFoundError:
        return None, "Model file 'lr_model.pkl' not found. Please ensure the model file is in the correct directory."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

model, model_error = load_model()

if model_error:
    st.error(f"⚠️ {model_error}")
    st.stop()

# Get the actual feature names from the trained model
@st.cache_resource
def get_model_features():
    try:
        if hasattr(model, 'feature_names_in_'):
            return model.feature_names_in_.tolist()
        else:
            return feature_cols
    except Exception:
        return feature_cols

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

model_features = get_model_features()

def prepare_data_for_model(df, model_features):
    """Prepare encoded dataframe to match model's expected features"""
    df_model = df.copy()
    
    # Add missing columns with 0 values if any
    for col in model_features:
        if col not in df_model.columns:
            df_model[col] = 0
    
    # Select only the features the model expects, in the correct order
    df_model = df_model[model_features]
    
    return df_model

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 Breast Cancer Risk Prediction</h1>
    <p>Advanced Machine Learning for Breast Cancer Prognosis Assessment</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 👤 Individual Patient Risk Assessment")

st.markdown("""
<div class="info-card">
    <h4>🔍 Patient Information</h4>
    <p>Enter the patient's clinical and pathological information below to get a personalized 10-year mortality risk assessment.</p>
    <p><em>Note: This form uses simplified inputs. The encoded dataset contains many more features that will be set to default values.</em></p>
</div>
""", unsafe_allow_html=True)

# Patient input form
with st.form("patient_form"):
    st.markdown("#### 📝 Basic Demographics & Tumor Characteristics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider('👤 Age at Diagnosis', 20, 100, 50, help="Patient's age when diagnosed")
        tumor_size = st.slider('📏 Tumor Size (mm)', 0, 100, 20, help="Maximum tumor diameter in millimeters")
        
    with col2:
        lymph_nodes = st.slider('🔗 Positive Lymph Nodes', 0, 50, 1, help="Number of lymph nodes with cancer cells")
        mutation_count = st.slider('🧬 Mutation Count', 0, 500, 10, help="Total number of genetic mutations detected")
        
    with col3:
        npi = st.slider('📊 Nottingham Prognostic Index', 0.0, 10.0, 4.5, help="Combined prognostic score")
        
    st.markdown("#### 🔬 Treatment & Biomarkers")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        chemo = st.selectbox(
            '💊 Chemotherapy', 
            [0, 1], 
            format_func=lambda x: '✅ Yes' if x == 1 else '❌ No',
            help="Whether patient received chemotherapy treatment"
        )
        
    with col5:
        er_status = st.selectbox(
            '🧪 ER Status', 
            [0, 1], 
            format_func=lambda x: '🟢 Positive' if x == 1 else '🔴 Negative',
            help="Estrogen receptor status"
        )
        
    with col6:
        pr_status = st.selectbox(
            '🧪 PR Status', 
            [0, 1], 
            format_func=lambda x: '🟢 Positive' if x == 1 else '🔴 Negative',
            help="Progesterone receptor status"
        )
    
    # Prediction button
    predict_button = st.form_submit_button("🔍 Analyze Patient Risk", type="primary")

if predict_button:
    # Build input dictionary
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

    # Create DataFrame with all model features, filling missing ones with 0
    input_df = pd.DataFrame([input_data])
    input_df_prepared = prepare_data_for_model(input_df, model_features)
    
    # Make prediction
    with st.spinner("🤖 Analyzing patient data..."):
        try:
            prediction = model.predict(input_df_prepared)[0]
            probability = model.predict_proba(input_df_prepared)[0]
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.error("Please check that your input data matches the model's requirements.")
            st.stop()
    
    # Display results with enhanced visualization
    st.markdown("---")
    st.markdown("### 📋 Risk Assessment Results")
    
    risk_score = probability[1]
    
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-positive">
            <h3>⚠️ HIGH RISK ASSESSMENT</h3>
            <h2>Risk Score: {risk_score:.1%}</h2>
            <p>This patient has a <strong>high predicted risk</strong> for 10-year mortality based on the provided clinical parameters.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-negative">
            <h3>✅ LOW RISK ASSESSMENT</h3>
            <h2>Risk Score: {risk_score:.1%}</h2>
            <p>This patient has a <strong>low predicted risk</strong> for 10-year mortality based on the provided clinical parameters.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk visualization
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "10-Year Mortality Risk (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Patient summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Patient Summary")
        st.write(f"**Age:** {age} years")
        st.write(f"**Tumor Size:** {tumor_size} mm")
        st.write(f"**Positive Lymph Nodes:** {lymph_nodes}")
        st.write(f"**Chemotherapy:** {'Yes' if chemo else 'No'}")
        
    with col2:
        st.markdown("#### 🔬 Biomarker Profile")
        st.write(f"**ER Status:** {'Positive' if er_status else 'Negative'}")
        st.write(f"**PR Status:** {'Positive' if pr_status else 'Negative'}")
        st.write(f"**Mutation Count:** {mutation_count}")
        st.write(f"**NPI Score:** {npi:.1f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h4>🏥 Breast Cancer Risk Prediction</h4>
    <p>⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only. Always consult healthcare professionals for medical decisions.</p>
    <p><em>Powered by Machine Learning & Advanced Analytics</em></p>
</div>
""", unsafe_allow_html=True)
