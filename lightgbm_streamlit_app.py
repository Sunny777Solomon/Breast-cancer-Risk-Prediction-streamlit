import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(
    page_title="Breastcare Risk Analyzer",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to set background image
def add_bg_from_local():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1579154204601-01588f822e49?q=80&w=2070&auto=format&fit=crop");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        
        /* Overlay to make content readable */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(253, 242, 248, 0.88);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to add background
add_bg_from_local()

# Custom CSS for the elegant medical theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        letter-spacing: -0.5px;
    }
    
    /* Navigation - subtle and top right */
    .nav-container {
        position: fixed;
        top: 20px;
        right: 30px;
        z-index: 1000;
        display: flex;
        gap: 20px;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(8px);
        padding: 10px 20px;
        border-radius: 50px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid rgba(219, 39, 119, 0.2);
    }
    
    .nav-btn {
        background: transparent;
        border: none;
        font-size: 1rem;
        font-weight: 500;
        color: #831843;
        cursor: pointer;
        padding: 5px 15px;
        border-radius: 25px;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .nav-btn:hover {
        background: rgba(219, 39, 119, 0.1);
        color: #9d174d;
    }
    
    .nav-btn.active {
        background: #db2777;
        color: white;
    }
    
    /* Cards and containers */
    .elegant-card {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(8px);
    }
    
    /* Section titles - darker colors for visibility */
    .section-title {
        color: #831843 !important;
        border-bottom: 2px solid #fbcfe8;
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem;
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Hero section */
    .hero-container {
        background: rgba(253, 242, 248, 0.88);
        backdrop-filter: blur(12px);
        padding: 3rem 2rem;
        border-radius: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #831843 !important;
        margin-bottom: 1rem;
        line-height: 1.2;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.5);
    }
    
    .hero-subtitle {
        font-size: 1.8rem;
        color: #b91c6f !important;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ec4899, #db2777);
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem 2rem;
        border-radius: 2rem;
        border: none;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(236, 72, 153, 0.4);
    }
    
    /* Risk result styling */
    .risk-high {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 2rem;
        border-radius: 1.5rem;
        border-left: 5px solid #ff4757;
        margin: 1rem 0;
        text-align: center;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 1.5rem;
        border-left: 5px solid #2ed573;
        margin: 1rem 0;
        text-align: center;
    }
    
    .risk-percentage {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
        color: #1e1e1e;
    }
    
    /* Form styling */
    .form-container {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 1.5rem;
        padding: 2rem;
    }
    
    /* Content text - darker for better readability */
    .content-text {
        line-height: 1.8;
        font-size: 1.1rem;
        color: #2d3748 !important;
    }
    
    ul.content-text {
        padding-left: 2rem;
    }
    
    ul.content-text li {
        margin-bottom: 0.75rem;
        color: #2d3748;
    }
    
    ul.content-text li strong {
        color: #831843;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.9);
        border-top: 1px solid #fbcfe8;
        margin-top: 3rem;
        color: #4a4a4a;
    }
    
    /* Loading animation */
    .stProgress > div > div {
        background-color: #db2777;
    }
    
    /* Spacing for content to account for fixed nav */
    .content-wrapper {
        margin-top: 80px;
    }
</style>
""", unsafe_allow_html=True)

# Load the Random Forest model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf.pkl')
        return model, None
    except FileNotFoundError:
        return None, "Model file 'rf.pkl' not found. Please ensure the model file is in the correct directory."
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
    """Prepare encoded dataframe to match model's expected features EXACTLY"""
    df_model = pd.DataFrame(0, index=df.index, columns=model_features)
    
    # Map the input features to the correct column names
    feature_mapping = {
        'Age at Diagnosis': 'Age at Diagnosis',
        'Lymph nodes examined positive': 'Lymph nodes examined positive',
        'Mutation Count': 'Mutation Count',
        'Nottingham prognostic index': 'Nottingham prognostic index',
        'Tumor Size': 'Tumor Size',
        'Chemotherapy': 'Chemotherapy',
        'ER Status': 'ER Status',
        'PR Status': 'PR Status'
    }
    
    for input_col, model_col in feature_mapping.items():
        if input_col in df.columns and model_col in df_model.columns:
            df_model[model_col] = df[input_col].values
    
    return df_model

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'probability_result' not in st.session_state:
    st.session_state.probability_result = None
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

# Navigation
st.markdown("""
<div class="nav-container">
    <button class="nav-btn" onclick="window.location.href='?nav=home'">🏠 Home</button>
    <button class="nav-btn" onclick="window.location.href='?nav=analyze'">🔬 Analyze</button>
</div>
""", unsafe_allow_html=True)

# Handle navigation via query params
query_params = st.query_params
if 'nav' in query_params:
    if query_params['nav'] == 'home':
        st.session_state.page = 'home'
    elif query_params['nav'] == 'analyze':
        st.session_state.page = 'analyze'
        st.session_state.form_submitted = False

# Home Page
if st.session_state.page == 'home':
    st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Breasts they could use your support</h1>
        <p class="hero-subtitle">Early awareness saves lives.</p>
    </div>
    
    <div class="elegant-card">
        <h2 class="section-title">🎗️ Understanding Breast Cancer</h2>
        <p class="content-text">
            Breast cancer is the most common cancer diagnosed in women and the second most common cause of death from cancer among women worldwide. The breasts are paired glands of variable size and density that lie superficial to the pectoralis major muscle. They contain milk-producing cells arranged in lobules; multiple lobules are aggregated into lobes with interspersed fat. Milk and other secretions are produced in acini and extruded through lactiferous ducts that exit at the nipple. Breasts are anchored to the underlying muscular fascia by Cooper ligaments, which support the breast. Breast cancer most commonly arises in the ductal epithelium (ie, ductal carcinoma) but can also develop in the breast lobules (ie, lobular carcinoma). Several risk factors for breast cancer have been well described. In Western countries, screening programs have succeeded in identifying most breast cancers through screening rather than due to symptoms. However, in much of the developing world, a breast mass or abnormal nipple discharge is often the presenting symptom. Breast cancer is diagnosed through physical examination, breast imaging, and tissue biopsy. Treatment options include surgery, chemotherapy, radiation, hormonal therapy, and, more recently, immunotherapy. Factors such as histology, stage, tumor markers, and genetic abnormalities guide individualized treatment decisions.
        </p>

        <h2 class="section-title" style="margin-top: 2.5rem;">⚠️ Breast Cancer Risk Factors</h2>
        <p class="content-text">
            Identifying factors associated with an increased incidence of breast cancer development is important in general health screening for women. Risk factors for breast cancer include:
        </p>
        <ul class="content-text">
            <li><strong>Age:</strong> The age-adjusted incidence of breast cancer continues to increase with the advancing age of the female population.</li>
            <li><strong>Gender:</strong> Most breast cancers occur in women.</li>
            <li><strong>Personal history:</strong> A history of cancer in one breast increases the likelihood of a second primary cancer in the contralateral breast.</li>
            <li><strong>Histologic:</strong> Histologic abnormalities diagnosed by breast biopsy constitute an essential category of breast cancer risk factors. These abnormalities include lobular carcinoma in situ (LCIS) and proliferative changes with atypia.</li>
            <li><strong>Family history and genetic mutations:</strong> First-degree relatives of patients with breast cancer have a 2-fold to 3-fold excess risk for the development of the disease. Genetic factors cause 5% to 10% of all breast cancer cases but may account for 25% of cases in women younger than 30 years. BRCA1 and BRCA2 are the most important genes responsible for increased breast cancer susceptibility.</li>
            <li><strong>Reproductive:</strong> Reproductive milestones that increase a woman's lifetime estrogen exposure are thought to increase breast cancer risk. These include the onset of menarche before age 12, first live childbirth after age 30 years, nulliparity, and menopause after the age of 55.</li>
            <li><strong>Exogenous hormone use:</strong> Therapeutic or supplemental estrogen and progesterone are taken for various conditions, with the most common scenarios being contraception in premenopausal women and hormone replacement therapy in postmenopausal women.</li>
            <li><strong>Other:</strong> Radiation, environmental exposures, obesity, and excessive alcohol consumption are some other factors that are associated with an increased risk of breast cancer.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Centered button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎗️ Start Your Risk Analysis", use_container_width=True):
            st.session_state.page = 'analyze'
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Analyze Page
elif st.session_state.page == 'analyze':
    st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)
    
    st.markdown("""
    <h2 style="font-family: 'Playfair Display', serif; font-size: 3rem; color: #831843; text-align: center; margin: 2rem 0;">
        10-Year Mortality Risk Assessment
    </h2>
    """, unsafe_allow_html=True)
    
    # Input form
    if not st.session_state.form_submitted:
        with st.form("patient_form"):
            st.markdown('<div class="form-container">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Demographics & Tumor Characteristics")
                age = st.slider('👤 Age at Diagnosis', 20, 100, 50, help="Patient's age when diagnosed")
                tumor_size = st.slider('📏 Tumor Size (mm)', 0, 150, 20, help="Maximum tumor diameter in millimeters")
                lymph_nodes = st.slider('🔗 Positive Lymph Nodes', 0, 50, 1, help="Number of lymph nodes with cancer cells")
                
            with col2:
                st.markdown("#### 🔬 Clinical Parameters")
                mutation_count = st.slider('🧬 Mutation Count', 0, 200, 10, help="Total number of genetic mutations detected")
                npi = st.slider('📊 Nottingham Prognostic Index', 1.0, 10.0, 4.5, 0.01, help="Combined prognostic score")
            
            st.markdown("#### 💊 Treatment & Biomarkers")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                chemo = st.selectbox(
                    'Chemotherapy', 
                    [0, 1], 
                    format_func=lambda x: '✅ Yes' if x == 1 else '❌ No',
                    help="Whether patient received chemotherapy treatment"
                )
                
            with col4:
                er_status = st.selectbox(
                    'ER Status', 
                    [0, 1], 
                    format_func=lambda x: '🟢 Positive' if x == 1 else '🔴 Negative',
                    help="Estrogen receptor status"
                )
                
            with col5:
                pr_status = st.selectbox(
                    'PR Status', 
                    [0, 1], 
                    format_func=lambda x: '🟢 Positive' if x == 1 else '🔴 Negative',
                    help="Progesterone receptor status"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Prediction button
            submitted = st.form_submit_button("🔍 Analyze Patient Risk", use_container_width=True)
            
            if submitted:
                # Store form data
                st.session_state.form_data = {
                    'Age at Diagnosis': age,
                    'Lymph nodes examined positive': lymph_nodes,
                    'Mutation Count': mutation_count,
                    'Nottingham prognostic index': npi,
                    'Tumor Size': tumor_size,
                    'Chemotherapy': chemo,
                    'ER Status': er_status,
                    'PR Status': pr_status
                }
                
                # Create DataFrame with all model features
                input_df = pd.DataFrame([st.session_state.form_data])
                input_df_prepared = prepare_data_for_model(input_df, model_features)
                
                # Debug info (optional - can remove in production)
                st.write("Model expects these features:", list(model_features[:10]) + ["..."])
                st.write("Input shape:", input_df_prepared.shape)
                
                # Show loading animation
                with st.spinner("🤖 Analyzing patient data..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    try:
                        prediction = model.predict(input_df_prepared)[0]
                        probability = model.predict_proba(input_df_prepared)[0]
                        
                        st.session_state.prediction_result = prediction
                        st.session_state.probability_result = probability
                        st.session_state.form_submitted = True
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                        st.error("Feature mismatch. Debug info below:")
                        st.write("Input features:", list(input_df_prepared.columns))
                        st.write("Model features:", list(model_features))
    
    # Results display
    else:
        prediction = st.session_state.prediction_result
        probability = st.session_state.probability_result
        risk_score = probability[1] * 100
        form_data = st.session_state.form_data
        
        # Risk result display
        if prediction == 1:
            st.markdown(f"""
            <div class="risk-high">
                <h2 style="color: #ff4757; margin-bottom: 1rem;">⚠️ HIGH 10-Year Mortality Risk</h2>
                <div class="risk-percentage">{risk_score:.1f}%</div>
                <p style="font-size: 1.2rem; color: #2d3748;">Please consult a doctor immediately.</p>
                <p style="margin-top: 1rem; color: #2d3748;">A {risk_score:.1f}% predicted risk indicates the importance of prompt medical evaluation and possible early intervention.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <h2 style="color: #2ed573; margin-bottom: 1rem;">✅ LOW 10-Year Mortality Risk</h2>
                <div class="risk-percentage">{risk_score:.1f}%</div>
                <p style="font-size: 1.2rem; color: #2d3748;">Keep going — you're doing wonderfully ❤️</p>
                <p style="margin-top: 1rem; color: #2d3748;">Continue with regular screenings and a healthy lifestyle. This result is very encouraging.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk visualization
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "10-Year Mortality Risk (%)", 'font': {'size': 24, 'color': '#831843'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#831843'},
                'bar': {'color': "#db2777"},
                'steps': [
                    {'range': [0, 25], 'color': "#a8edea"},
                    {'range': [25, 50], 'color': "#fed6e3"},
                    {'range': [50, 75], 'color': "#ff9a9e"},
                    {'range': [75, 100], 'color': "#fecfef"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        
        fig_gauge.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font={'color': "#831843"})
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Patient summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Patient Summary")
            st.write(f"**Age:** {form_data['Age at Diagnosis']} years")
            st.write(f"**Tumor Size:** {form_data['Tumor Size']} mm")
            st.write(f"**Positive Lymph Nodes:** {form_data['Lymph nodes examined positive']}")
            st.write(f"**Chemotherapy:** {'Yes' if form_data['Chemotherapy'] else 'No'}")
            
        with col2:
            st.markdown("#### 🔬 Biomarker Profile")
            st.write(f"**ER Status:** {'Positive' if form_data['ER Status'] else 'Negative'}")
            st.write(f"**PR Status:** {'Positive' if form_data['PR Status'] else 'Negative'}")
            st.write(f"**Mutation Count:** {form_data['Mutation Count']}")
            st.write(f"**NPI Score:** {form_data['Nottingham prognostic index']:.1f}")
        
        # Analyze another button
        if st.button("🔄 Analyze Another Case", use_container_width=True):
            st.session_state.form_submitted = False
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p style="font-size: 2rem; margin-bottom: 1rem;">🎗️</p>
    <p><strong>This is an educational awareness tool only — always consult a qualified healthcare professional for medical advice.</strong></p>
    <p style="margin-top: 1rem; color: #db2777;">Powered by Machine Learning & Advanced Analytics</p>
</div>
""", unsafe_allow_html=True)

# Add Font Awesome for icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)
