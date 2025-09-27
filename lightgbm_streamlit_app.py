import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Risk Prediction",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Professional CSS styling with medical theme
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0066CC 0%, #004A99 100%);
        padding: 3rem 2rem;
        border-radius: 0 0 30px 30px;
        color: white;
        text-align: center;
        margin: -1rem -1rem 3rem -1rem;
        box-shadow: 0 10px 30px rgba(0, 102, 204, 0.15);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 102, 204, 0.1);
    }
    
    .info-card h4 {
        color: #004A99;
        font-weight: 600;
        margin-bottom: 0.8rem;
        font-size: 1.2rem;
    }
    
    .info-card p {
        color: #495057;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Form section headers */
    .form-section {
        background: #f8f9fa;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border-left: 4px solid #0066CC;
    }
    
    .form-section h4 {
        color: #004A99;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Result cards */
    .result-high-risk {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%);
        padding: 2rem;
        border-radius: 16px;
        border-left: 5px solid #dc3545;
        box-shadow: 0 4px 20px rgba(220, 53, 69, 0.1);
        margin: 2rem 0;
    }
    
    .result-low-risk {
        background: linear-gradient(135deg, #f0fff4 0%, #dcffe4 100%);
        padding: 2rem;
        border-radius: 16px;
        border-left: 5px solid #28a745;
        box-shadow: 0 4px 20px rgba(40, 167, 69, 0.1);
        margin: 2rem 0;
    }
    
    .result-high-risk h3, .result-low-risk h3 {
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .result-high-risk h3 {
        color: #dc3545;
    }
    
    .result-low-risk h3 {
        color: #28a745;
    }
    
    .risk-score {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 1rem 0;
        color: #212529;
    }
    
    /* Patient summary styling */
    .summary-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 2rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .summary-section h4 {
        color: #004A99;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    .summary-item {
        padding: 0.4rem 0;
        color: #495057;
        font-size: 0.95rem;
    }
    
    .summary-item strong {
        color: #212529;
        font-weight: 500;
    }
    
    /* Disclaimer styling */
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
        color: #856404;
    }
    
    .disclaimer h5 {
        color: #856404;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0066CC 0%, #004A99 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 102, 204, 0.3);
    }
    
    /* Input field styling */
    .stSlider > div > div > div > div {
        background: #0066CC;
    }
    
    .stSelectbox > div > div > select {
        border-color: #e9ecef;
        font-size: 0.95rem;
    }
    
    /* Metric display */
    .metric-display {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load the trained model
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

# Get model features
@st.cache_resource
def get_model_features():
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
    return feature_cols

model_features = get_model_features()

# Header
st.markdown("""
<div class="main-header">
    <h1>Breast Cancer Risk Prediction</h1>
    <p>Advanced AI-Powered Clinical Decision Support System</p>
</div>
""", unsafe_allow_html=True)

# Patient Information Card
st.markdown("""
<div class="info-card">
    <h4>Patient Risk Assessment</h4>
    <p>This clinical decision support tool uses advanced machine learning to assess 10-year mortality risk in breast cancer patients. 
    Please enter the patient's clinical and pathological information below for a comprehensive risk evaluation.</p>
</div>
""", unsafe_allow_html=True)

# Patient input form
with st.form("patient_assessment_form"):
    # Demographics Section
    st.markdown("""
    <div class="form-section">
        <h4>Demographics & Tumor Characteristics</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider(
            'Age at Diagnosis (years)', 
            20, 100, 50,
            help="Patient's age at initial diagnosis"
        )
        
        tumor_size = st.slider(
            'Tumor Size (mm)', 
            0, 100, 20,
            help="Maximum tumor diameter"
        )
    
    with col2:
        lymph_nodes = st.slider(
            'Positive Lymph Nodes', 
            0, 50, 1,
            help="Number of lymph nodes with metastasis"
        )
        
        mutation_count = st.slider(
            'Mutation Count', 
            0, 500, 10,
            help="Total genetic mutations detected"
        )
    
    with col3:
        npi = st.slider(
            'Nottingham Prognostic Index', 
            0.0, 10.0, 4.5, 0.1,
            help="Combined prognostic score"
        )
        
        tumor_stage = st.selectbox(
            'Tumor Stage',
            ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'],
            help="Clinical tumor stage"
        )
    
    # Treatment Section
    st.markdown("""
    <div class="form-section">
        <h4>Treatment & Biomarkers</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col4, col5, col6, col7 = st.columns(4)
    
    with col4:
        chemo = st.selectbox(
            'Chemotherapy', 
            ['No', 'Yes'],
            help="Chemotherapy treatment status"
        )
    
    with col5:
        er_status = st.selectbox(
            'ER Status', 
            ['Negative', 'Positive'],
            help="Estrogen receptor status"
        )
    
    with col6:
        pr_status = st.selectbox(
            'PR Status', 
            ['Negative', 'Positive'],
            help="Progesterone receptor status"
        )
    
    with col7:
        her2_status = st.selectbox(
            'HER2 Status',
            ['Negative', 'Positive'],
            help="HER2 receptor status"
        )
    
    col8, col9, col10, col11 = st.columns(4)
    
    with col8:
        hormone_therapy = st.selectbox(
            'Hormone Therapy',
            ['No', 'Yes'],
            help="Hormone therapy treatment"
        )
    
    with col9:
        radio_therapy = st.selectbox(
            'Radiotherapy',
            ['No', 'Yes'],
            help="Radiotherapy treatment"
        )
    
    with col10:
        surgery_type = st.selectbox(
            'Surgery Type',
            ['Lumpectomy', 'Mastectomy'],
            help="Type of surgical intervention"
        )
    
    with col11:
        grade = st.selectbox(
            'Histologic Grade',
            ['Grade 1', 'Grade 2', 'Grade 3'],
            help="Tumor differentiation grade"
        )
    
    # Submit button
    submitted = st.form_submit_button("Generate Risk Assessment", type="primary", use_container_width=True)

# Process prediction when form is submitted
if submitted:
    # Convert inputs to model format
    input_data = {
        'Age at Diagnosis': age,
        'Lymph nodes examined positive': lymph_nodes,
        'Mutation Count': mutation_count,
        'Nottingham prognostic index': npi,
        'Tumor Size': tumor_size,
        'Chemotherapy': 1 if chemo == 'Yes' else 0,
        'ER Status': 1 if er_status == 'Positive' else 0,
        'PR Status': 1 if pr_status == 'Positive' else 0,
        'HER2 Status': 1 if her2_status == 'Positive' else 0,
        'Hormone Therapy': 1 if hormone_therapy == 'Yes' else 0,
        'Radio Therapy': 1 if radio_therapy == 'Yes' else 0,
        'Type of Breast Surgery_Mastectomy': 1 if surgery_type == 'Mastectomy' else 0
    }
    
    # Handle tumor stage encoding
    for i in range(1, 5):
        input_data[f'Tumor Stage_{i}.0'] = 1 if tumor_stage == f'Stage {i}' else 0
    
    # Handle grade encoding
    if grade == 'Grade 2':
        input_data['Neoplasm Histologic Grade_2.0'] = 1
    elif grade == 'Grade 3':
        input_data['Neoplasm Histologic Grade_3.0'] = 1
    
    # Fill remaining features with 0
    for col in model_features:
        if col not in input_data:
            input_data[col] = 0
    
    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([input_data])[model_features]
    
    # Make prediction
    with st.spinner("Analyzing patient data..."):
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            risk_score = probability[1]
            
            # Display results
            st.markdown("---")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="result-high-risk">
                    <h3>High Risk Assessment</h3>
                    <div class="risk-score">{risk_score:.1%}</div>
                    <p>The patient presents with a high predicted risk for 10-year mortality based on the clinical parameters provided. 
                    Close monitoring and aggressive treatment strategies should be considered.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low-risk">
                    <h3>Low Risk Assessment</h3>
                    <div class="risk-score">{risk_score:.1%}</div>
                    <p>The patient presents with a low predicted risk for 10-year mortality based on the clinical parameters provided. 
                    Standard treatment protocols with regular follow-up are recommended.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk visualization gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "10-Year Mortality Risk", 'font': {'size': 20, 'color': '#004A99'}},
                number = {'suffix': "%", 'font': {'size': 40, 'color': '#212529'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#495057"},
                    'bar': {'color': "#0066CC", 'thickness': 0.8},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#dee2e6",
                    'steps': [
                        {'range': [0, 25], 'color': '#d4edda'},
                        {'range': [25, 50], 'color': '#fff3cd'},
                        {'range': [50, 75], 'color': '#f8d7da'},
                        {'range': [75, 100], 'color': '#f5c6cb'}
                    ],
                    'threshold': {
                        'line': {'color': "#dc3545", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                height=350,
                margin=dict(t=50, b=50, l=50, r=50),
                paper_bgcolor="white",
                font={'family': "Inter, sans-serif"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Patient Summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="summary-section">
                    <h4>Clinical Parameters</h4>
                    <div class="summary-item"><strong>Age:</strong> """ + str(age) + """ years</div>
                    <div class="summary-item"><strong>Tumor Size:</strong> """ + str(tumor_size) + """ mm</div>
                    <div class="summary-item"><strong>Positive Lymph Nodes:</strong> """ + str(lymph_nodes) + """</div>
                    <div class="summary-item"><strong>NPI Score:</strong> """ + f"{npi:.1f}" + """</div>
                    <div class="summary-item"><strong>Tumor Stage:</strong> """ + tumor_stage + """</div>
                    <div class="summary-item"><strong>Histologic Grade:</strong> """ + grade + """</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="summary-section">
                    <h4>Treatment & Biomarkers</h4>
                    <div class="summary-item"><strong>ER Status:</strong> """ + er_status + """</div>
                    <div class="summary-item"><strong>PR Status:</strong> """ + pr_status + """</div>
                    <div class="summary-item"><strong>HER2 Status:</strong> """ + her2_status + """</div>
                    <div class="summary-item"><strong>Chemotherapy:</strong> """ + chemo + """</div>
                    <div class="summary-item"><strong>Hormone Therapy:</strong> """ + hormone_therapy + """</div>
                    <div class="summary-item"><strong>Radiotherapy:</strong> """ + radio_therapy + """</div>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating prediction: {str(e)}")

# Medical Disclaimer
st.markdown("""
<div class="disclaimer">
    <h5>⚠️ Medical Disclaimer</h5>
    <p>This prediction tool is designed for research and educational purposes only. It should not be used as the sole basis for clinical decision-making. 
    All results must be interpreted by qualified healthcare professionals in conjunction with comprehensive clinical assessment, additional diagnostic tests, 
    and individual patient factors. Always consult with oncology specialists for treatment planning and patient management.</p>
</div>
""", unsafe_allow_html=True)
