import streamlit as st
import pandas as pd
import joblib
import requests
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="METABRIC Mortality Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
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
    
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
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
    
    .sidebar-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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
            # Fallback to predefined feature list if model doesn't have feature_names_in_
            return feature_cols
    except Exception as e:
        st.error(f"Error getting model features: {e}")
        return feature_cols

# Get actual model features
model_features = get_model_features()
st.sidebar.info(f"Model expects {len(model_features)} features")

# Debug information
if st.sidebar.checkbox("🔍 Show Model Features (Debug)"):
    st.sidebar.write("Expected features:")
    for i, feat in enumerate(model_features):
        st.sidebar.write(f"{i+1}. {feat}")

# ACTUAL COLUMNS FROM YOUR METABRIC CSV FILE
actual_csv_columns = [
    'Patient ID', 'Age at Diagnosis', 'Type of Breast Surgery', 'Cancer Type',
    'Cancer Type Detailed', 'Cellularity', 'Chemotherapy', 'Pam50 + Claudin-low subtype',
    'Cohort', 'ER status measured by IHC', 'ER Status', 'Neoplasm Histologic Grade',
    'HER2 status measured by SNP6', 'HER2 Status', 'Tumor Other Histologic Subtype',
    'Hormone Therapy', 'Inferred Menopausal State', 'Integrative Cluster',
    'Primary Tumor Laterality', 'Lymph nodes examined positive', 'Mutation Count',
    'Nottingham prognostic index', 'Oncotree Code', 'Overall Survival (Months)',
    'Overall Survival Status', 'PR Status', 'Radio Therapy', 'Relapse Free Status (Months)',
    'Relapse Free Status', 'Sex', '3-Gene classifier subtype', 'Tumor Size',
    'Tumor Stage', "Patient's Vital Status"
]

# Define all features used in training (your original model features)
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

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 METABRIC 10-Year Mortality Prediction System</h1>
    <p>Advanced Machine Learning for Breast Cancer Prognosis Assessment</p>
    <p><em>Professional Medical Decision Support Tool</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-info">
        <h3>📊 About METABRIC</h3>
        <p>The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC) dataset contains clinical and genomic data for breast cancer patients.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-info">
        <h3>🔬 Model Information</h3>
        <p><strong>Algorithm:</strong> Logistic Regression<br>
        <strong>Features:</strong> 56 clinical & genomic variables<br>
        <strong>Prediction:</strong> 10-year mortality risk</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-info">
        <h3>⚠️ Medical Disclaimer</h3>
        <p>This tool is for research and educational purposes only. Always consult healthcare professionals for medical decisions.</p>
    </div>
    """, unsafe_allow_html=True)

# GitHub CSV URL - Using the encoded version
github_csv_url = "https://raw.githubusercontent.com/Sunny777Solomon/Breast-cancer-Risk-Prediction-streamlit/refs/heads/main/METABRIC_encoded.csv"

@st.cache_data
def load_data_from_github(url):
    """Load CSV data from GitHub with caching"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content)
        
        return df, None
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching data from GitHub: {str(e)}"
    except pd.errors.EmptyDataError:
        return None, "The CSV file appears to be empty"
    except Exception as e:
        return None, f"Error processing CSV: {str(e)}"

def prepare_data_for_model(df, model_features):
    """Prepare encoded dataframe to match model's expected features"""
    # Create a copy to avoid modifying original
    df_model = df.copy()
    
    # Since this is already encoded, we just need to ensure feature alignment
    # Add missing columns with 0 values if any
    missing_features = []
    for col in model_features:
        if col not in df_model.columns:
            df_model[col] = 0
            missing_features.append(col)
    
    if missing_features:
        st.sidebar.warning(f"⚠️ Added {len(missing_features)} missing features with default value 0")
        if st.sidebar.checkbox("Show missing features"):
            for feat in missing_features:
                st.sidebar.text(f"• {feat}")
    
    # Select only the features the model expects, in the correct order
    try:
        df_model = df_model[model_features]
        st.sidebar.success(f"✅ Successfully aligned {len(model_features)} features")
    except KeyError as e:
        st.sidebar.error(f"❌ Feature alignment error: {e}")
        return None
    
    return df_model

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📊 Dataset Analysis", "👤 Single Patient Prediction", "📈 Model Insights"])

with tab1:
    st.markdown("### 📊 METABRIC Dataset Analysis")
    
    # Load data from GitHub
    with st.spinner("🔄 Loading dataset from GitHub repository..."):
        df, error = load_data_from_github(github_csv_url)
    
    if error:
        st.error(f"❌ {error}")
        st.info("💡 Please check your GitHub URL and ensure the CSV file is publicly accessible.")
    else:
        st.success("✅ Encoded dataset loaded successfully from GitHub!")
        
        # Show data info
        st.info(f"📊 Loaded {len(df)} patients with {len(df.columns)} encoded features")
        
        # Display first few columns to verify encoding
        if st.checkbox("🔍 Preview encoded data"):
            st.write("First 5 rows and 10 columns:")
            st.dataframe(df.iloc[:5, :10])
        
        # Prepare data for model prediction
        df_features = prepare_data_for_model(df, model_features)
        
        if df_features is None:
            st.error("❌ Failed to prepare data for model. Please check feature alignment.")
            st.stop()
        
        # Show feature matching info
        st.success(f"✅ Data prepared: {len(df_features.columns)} features aligned with model")

        # Make predictions
        with st.spinner("🤖 Running predictions on encoded dataset..."):
            try:
                predictions = model.predict(df_features)
                probabilities = model.predict_proba(df_features)
                
                st.success(f"🎯 Predictions completed for {len(predictions)} patients")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.error("Model feature mismatch detected. Check the debug panel in sidebar.")
                st.stop()
            
        df['Predicted 10-Year Mortality'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
        df['Mortality Probability'] = [prob[1] for prob in probabilities]

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        mortality_counts = df['Predicted 10-Year Mortality'].value_counts()
        total_patients = len(df)
        
        with col1:
            st.metric(
                "👥 Total Patients", 
                f"{total_patients:,}",
                help="Total number of patients in the dataset"
            )
        
        with col2:
            low_risk = mortality_counts.get('Low Risk', 0)
            st.metric(
                "✅ Low Risk", 
                f"{low_risk:,}",
                f"{(low_risk/total_patients*100):.1f}%"
            )
        
        with col3:
            high_risk = mortality_counts.get('High Risk', 0)
            st.metric(
                "⚠️ High Risk", 
                f"{high_risk:,}",
                f"{(high_risk/total_patients*100):.1f}%"
            )
            
        with col4:
            avg_prob = df['Mortality Probability'].mean()
            st.metric(
                "📊 Avg Risk Score", 
                f"{avg_prob:.3f}",
                help="Average mortality probability across all patients"
            )

        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution pie chart
            fig_pie = px.pie(
                values=mortality_counts.values, 
                names=mortality_counts.index,
                title="Risk Distribution",
                color_discrete_map={'Low Risk': '#2ed573', 'High Risk': '#ff4757'}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Probability distribution histogram
            fig_hist = px.histogram(
                df, 
                x='Mortality Probability', 
                nbins=30,
                title="Mortality Probability Distribution",
                color_discrete_sequence=['#667eea']
            )
            fig_hist.update_layout(
                xaxis_title="Mortality Probability",
                yaxis_title="Number of Patients"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Data table with enhanced display
        st.markdown("### 📋 Detailed Results")
        
        # Add risk level colors to dataframe display
        def color_risk(val):
            color = '#ffcccb' if val == 'High Risk' else '#d4edda'
            return f'background-color: {color}'
        
        display_df = df[['Predicted 10-Year Mortality', 'Mortality Probability'] + 
                       ['Age at Diagnosis', 'Tumor Size', 'Lymph nodes examined positive'][:3]]
        
        st.dataframe(
            display_df.style.applymap(color_risk, subset=['Predicted 10-Year Mortality']),
            use_container_width=True
        )
        
        # Download functionality
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Results (CSV)",
            data=csv,
            file_name=f"metabric_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Download the complete dataset with predictions"
        )

with tab2:
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
        
        col4, col5, col6, col7 = st.columns(4)
        
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
            
        with col7:
            st.write("")  # Spacer
        
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

with tab3:
    st.markdown("### 📈 Model Performance & Insights")
    
    st.markdown("""
    <div class="info-card">
        <h4>🤖 Machine Learning Model Details</h4>
        <p>This section provides insights into the logistic regression model used for predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Model Specifications")
        st.write("- **Algorithm:** Logistic Regression")
        st.write("- **Total Features:** 56")
        st.write("- **Target:** 10-year mortality (binary)")
        st.write("- **Data Source:** METABRIC dataset")
        st.write("- **Clinical Variables:** Age, tumor size, lymph nodes")
        st.write("- **Molecular Variables:** ER, PR, HER2 status")
        st.write("- **Genomic Variables:** Mutation count, subtypes")
        
    with col2:
        st.markdown("#### ⚠️ Important Considerations")
        st.warning("""
        **Medical Disclaimer:** This model is designed for research and educational purposes. 
        Clinical decisions should always involve qualified healthcare professionals and 
        consider additional factors not included in this model.
        """)
        
        st.info("""
        **Model Limitations:**
        - Based on historical data
        - May not account for treatment advances
        - Individual patient factors may vary
        - Requires clinical validation
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h4>🏥 METABRIC Mortality Prediction System</h4>
    <p>Developed for clinical research and educational purposes • Always consult healthcare professionals</p>
    <p><em>Powered by Machine Learning & Advanced Analytics</em></p>
</div>
""", unsafe_allow_html=True)
