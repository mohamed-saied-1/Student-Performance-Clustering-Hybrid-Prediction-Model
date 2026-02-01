import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import sys
from student_analysis import StudentPerformanceAnalyzer
from visualizations import StudentVisualizations
import plotly.express as px
import plotly.graph_objects as go

# Page configuration with dark theme
st.set_page_config(
    page_title="Student Performance Clustering & Hybrid Model",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* ===== MAIN THEME ===== */
    .stApp {
        background-color: #0F172A;
        color: #E2E8F0;
    }
    
    /* ===== HEADERS ===== */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #60A5FA, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #38BDF8;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid rgba(56, 189, 248, 0.3);
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .subsection-header {
        font-size: 1.3rem;
        color: #7DD3FC;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
        padding-left: 0.5rem;
        border-left: 4px solid #7DD3FC;
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.2) !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6, #1D4ED8) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563EB, #1E40AF) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3) !important;
    }
    
    .upload-button {
        background: linear-gradient(135deg, #3B82F6, #1D4ED8);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s;
        width: 100%;
        margin-top: 1rem;
    }
    
    .upload-button:hover {
        background: linear-gradient(135deg, #2563EB, #1E40AF);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .secondary-button > button {
        background: linear-gradient(135deg, #64748B, #475569) !important;
    }
    
    .secondary-button > button:hover {
        background: linear-gradient(135deg, #475569, #334155) !important;
    }
    
    /* ===== CARDS ===== */
    .custom-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(100, 116, 139, 0.3);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        border-color: rgba(59, 130, 246, 0.5);
    }
    
    .upload-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px dashed #3B82F6;
        text-align: center;
        transition: all 0.3s;
    }
    
    .upload-card:hover {
        border-color: #60A5FA;
        background: linear-gradient(145deg, #1E40AF, #1E293B);
    }
    
    .info-card {
        background: rgba(30, 41, 59, 0.7);
        border-left: 5px solid #10B981;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: rgba(245, 158, 11, 0.1);
        border-left: 5px solid #F59E0B;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* ===== STATUS MESSAGES ===== */
    .status-success {
        padding: 1rem;
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10B981;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .status-warning {
        padding: 1rem;
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #F59E0B;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .status-error {
        padding: 1rem;
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #EF4444;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* ===== FIX FOR FORM ELEMENTS ===== */
    /* Fix for unrendered HTML codes - RADIO BUTTONS */
    .stRadio > div {
        flex-direction: column;
        align-items: stretch;
    }
    
    .stRadio > div > label {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
        text-align: left;
        transition: all 0.3s;
    }
    
    .stRadio > div > label:hover {
        background-color: #2D3748;
        border-color: #3B82F6;
    }
    
    .stRadio > div > label > div:first-child {
        display: none;
    }
    
    .stRadio > div > label > div:last-child {
        width: 100%;
        font-weight: 500;
        color: #E2E8F0;
    }
    
    .stRadio > div > label[data-testid="stRadioSelected"] {
        background-color: #3B82F6;
        border-color: #3B82F6;
        color: white;
    }
    
    /* Fix checkbox styling */
    .stCheckbox > label {
        color: #E2E8F0 !important;
        font-weight: 500;
    }
    
    .stCheckbox > label > div:first-child {
        background-color: #1E293B;
        border: 2px solid #475569;
    }
    
    .stCheckbox > label > div:first-child:hover {
        border-color: #3B82F6;
    }
    
    /* Fix file uploader styling */
    .stFileUploader > label {
        color: transparent !important;
    }
    
    .stFileUploader > label > div:first-child {
        display: none;
    }
    
    .stFileUploader > section {
        border: 2px dashed #475569;
        border-radius: 10px;
        background-color: #1E293B;
        padding: 2rem;
        transition: all 0.3s;
        min-height: 150px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .stFileUploader > section:hover {
        border-color: #3B82F6;
        background-color: #1E293B;
    }
    
    .stFileUploader > section > div > span {
        color: #94A3B8 !important;
        text-align: center;
        display: block;
    }
    
    .stFileUploader > section > div > span:first-child {
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        color: #E2E8F0 !important;
    }
    
    /* Fix warning messages */
    .stAlert {
        background-color: rgba(245, 158, 11, 0.1) !important;
        border-left: 4px solid #F59E0B !important;
        border-radius: 4px !important;
        border: none !important;
        padding: 1rem !important;
    }
    
    .stAlert > div {
        color: #E2E8F0 !important;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1E293B;
        padding: 8px;
        border-radius: 10px;
        border: 1px solid rgba(100, 116, 139, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        border: none !important;
        color: #94A3B8 !important;
        font-weight: 500 !important;
        transition: all 0.3s !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(59, 130, 246, 0.1) !important;
        color: #CBD5E1 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3B82F6, #1D4ED8) !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(100, 116, 139, 0.3) !important;
        border-radius: 8px !important;
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(30, 41, 59, 0.9) !important;
        border-color: #3B82F6 !important;
    }
    
    /* ===== METRICS ===== */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #22D3EE !important; /* Bright cyan - vibrant and modern */
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    
    /* ===== DATAFRAMES ===== */
    .dataframe {
        background-color: #1E293B !important;
        border: 1px solid rgba(100, 116, 139, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .dataframe th {
        background-color: #334155 !important;
        color: #E2E8F0 !important;
        font-weight: 600 !important;
        border: none !important;
    }
    
    .dataframe td {
        color: #CBD5E1 !important;
        border-color: rgba(100, 116, 139, 0.3) !important;
    }
    
    /* ===== PROGRESS ===== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3B82F6, #60A5FA) !important;
    }
    
    /* ===== SIDEBAR NAVIGATION BUTTONS ===== */
    .sidebar-button {
        width: 100%;
        text-align: left;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        border: 1px solid #334155;
        background-color: #1E293B;
        color: #E2E8F0;
        font-weight: 500;
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .sidebar-button:hover {
        background-color: #2D3748;
        border-color: #3B82F6;
        transform: translateX(5px);
    }
    
    .sidebar-button.active {
        background-color: #3B82F6;
        border-color: #3B82F6;
        color: white;
    }
    
    /* ===== TOOLTIPS ===== */
    [data-testid="stTooltip"] {
        background-color: #1E293B !important;
        border: 1px solid rgba(100, 116, 139, 0.3) !important;
        color: #E2E8F0 !important;
    }
    
    /* ===== LISTS ===== */
    ul.custom-list {
        list-style-type: none;
        padding-left: 0;
    }
    
    ul.custom-list li {
        padding: 0.5rem 0;
        color: #CBD5E1;
        display: flex;
        align-items: flex-start;
    }
    
    ul.custom-list li:before {
        content: "✓";
        color: #10B981;
        font-weight: bold;
        margin-right: 10px;
    }
    
    /* ===== FIX FOR HTML RENDERING ===== */
    .stMarkdown {
        color: #CBD5E1 !important;
        line-height: 1.6 !important;
    }
    
    .stMarkdown strong {
        color: #60A5FA !important;
    }
    
    .stMarkdown em {
        color: #94A3B8 !important;
    }
    
    /* ===== GRID LAYOUT HELPERS ===== */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    /* ===== CUSTOM ALERTS ===== */
    .custom-alert {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid;
        background: rgba(30, 41, 59, 0.7);
    }
    
    .alert-success {
        border-left-color: #10B981;
        color: #A7F3D0;
    }
    
    .alert-warning {
        border-left-color: #F59E0B;
        color: #FDE68A;
    }
    
    .alert-error {
        border-left-color: #EF4444;
        color: #FCA5A5;
    }
    
    .alert-info {
        border-left-color: #3B82F6;
        color: #BFDBFE;
    }
    
    /* ===== CODE BLOCKS ===== */
    .stCodeBlock {
        background-color: #1E293B !important;
        border: 1px solid rgba(100, 116, 139, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* ===== FIX FOR PLOTLY CHARTS ===== */
    .js-plotly-plot {
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    /* ===== CUSTOM DIVIDERS ===== */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(100, 116, 139, 0.3), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'viz' not in st.session_state:
        st.session_state.viz = StudentVisualizations()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'clustering_done' not in st.session_state:
        st.session_state.clustering_done = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model_comparison_done' not in st.session_state:
        st.session_state.model_comparison_done = False
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None
    if 'current_menu' not in st.session_state:
        st.session_state.current_menu = "🏠 Home"
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    if 'n_clusters' not in st.session_state:
        st.session_state.n_clusters = 2
    if 'prediction_mode' not in st.session_state:
        st.session_state.prediction_mode = "single"  # "single" or "batch"

init_session_state()

def render_header(title, subtitle=None):
    """Render a styled header"""
    st.markdown(f'<h1 class="main-header">{title}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p style="text-align: center; color: #94A3B8; font-size: 1.1rem; margin-bottom: 2rem;">{subtitle}</p>', unsafe_allow_html=True)

def render_section_header(title):
    """Render a section header"""
    st.markdown(f'<h2 class="section-header">{title}</h2>', unsafe_allow_html=True)

def handle_file_upload(uploaded_file):
    """Handle uploaded CSV file"""
    try:
        with st.spinner("Processing uploaded file..."):
            # Read the uploaded file
            content = uploaded_file.getvalue().decode('utf-8')
            
            # Try different delimiters - first try semicolon (as in notebook)
            try:
                df = pd.read_csv(io.StringIO(content), delimiter=';')
            except:
                # If semicolon fails, try comma
                try:
                    df = pd.read_csv(io.StringIO(content), delimiter=',')
                except Exception as e:
                    st.error(f"Could not read CSV file. Error: {str(e)}")
                    return False
            
            # Check if required columns exist (minimum from notebook)
            required_cols = ['G1', 'G2', 'G3']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Uploaded file is missing required columns: {', '.join(missing_cols)}")
                st.info("Please upload a CSV file with at least G1, G2, and G3 columns.")
                return False
            
            # Initialize analyzer with uploaded data
            st.session_state.analyzer = StudentPerformanceAnalyzer(df=df)
            st.session_state.analyzer.preprocess_data()
            
            # Validate data structure
            is_valid, message = st.session_state.analyzer.validate_data_structure()
            if not is_valid:
                st.error(f"Data validation failed: {message}")
                st.session_state.analyzer = None
                return False
            
            st.session_state.data_loaded = True
            st.session_state.uploaded_file = uploaded_file
            st.session_state.uploaded_filename = uploaded_file.name
            
            # Reset other states
            st.session_state.clustering_done = False
            st.session_state.model_trained = False
            st.session_state.model_comparison_done = False
            
        return True
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        return False

def show_home():
    """Home page with data upload options"""
    render_header("Student Performance Analytics", "Clustering & Hybrid Machine Learning Dashboard")
    
    # Hero Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #60A5FA; margin-top: 0;">📊 Platform Overview</h3>
            <p style="color: #CBD5E1; line-height: 1.6;">
            This platform implements the exact machine learning pipeline from the research notebook. 
            It provides comprehensive analysis of student academic performance using advanced clustering 
            and predictive modeling techniques.
            </p>
            <p style="color: #94A3B8; font-weight: 600; margin-top: 1rem;">Exact Notebook Implementation:</p>
            <ul style="color: #94A3B8;">
                <li>✅ Same data preprocessing and outlier handling</li>
                <li>✅ Same K-means clustering with elbow method</li>
                <li>✅ Same hybrid RandomForest model with GridSearchCV</li>
                <li>✅ Same model comparison methodology</li>
                <li>✅ Same feature engineering and target encoding</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=150)
    
    # Get Started Section
    render_section_header("🚀 Get Started")
    
    if not st.session_state.data_loaded:
        # Main upload section in Home tab
        st.markdown("""
        <div class="upload-card">
            <h3 style="color: #60A5FA; margin-top: 0;">📁 Upload Your Student Data</h3>
            <p style="color: #CBD5E1; line-height: 1.6;">
            Upload your student performance dataset in CSV format. The system expects data similar to 
            the original "student-mat.csv" dataset used in the research notebook.
            </p>
            <p style="color: #94A3B8; font-weight: 600;">Requirements:</p>
            <ul style="color: #94A3B8;">
                <li><strong>Format:</strong> CSV file (.csv)</li>
                <li><strong>Delimiter:</strong> Semicolon (;) or comma (,)</li>
                <li><strong>Required Columns:</strong> G1, G2, G3 (grades)</li>
                <li><strong>Recommended:</strong> Additional student attributes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your student data CSV file",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Show file info
            st.markdown(f"""
            <div style="background: #1E293B; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">📄</span>
                    <div>
                        <div style="font-weight: 600; color: #E2E8F0;">{uploaded_file.name}</div>
                        <div style="font-size: 0.9rem; color: #94A3B8;">
                            Size: {uploaded_file.size / 1024:.1f} KB
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Preview option
            if st.checkbox("📋 Preview uploaded data"):
                try:
                    content = uploaded_file.getvalue().decode('utf-8')
                    # Reset file pointer
                    uploaded_file.seek(0)
                    try:
                        preview_df = pd.read_csv(io.StringIO(content), delimiter=';', nrows=10)
                    except:
                        preview_df = pd.read_csv(io.StringIO(content), delimiter=',', nrows=10)
                    st.dataframe(preview_df)
                    st.write(f"**Shape:** {preview_df.shape[0]} rows × {preview_df.shape[1]} columns")
                except Exception as e:
                    st.warning(f"Could not preview file: {str(e)}")
            
            # Upload button
            if st.button("🚀 Upload & Process File", type="primary", use_container_width=True):
                if handle_file_upload(uploaded_file):
                    st.success(f"✅ {uploaded_file.name} uploaded and processed successfully!")
                    st.rerun()
    else:
        # Data is loaded - show stats
        analyzer = st.session_state.analyzer
        stats = analyzer.get_academic_summary()
        
        st.markdown(f"""
        <div class="status-success">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="color: #10B981; margin: 0;">✅ Data Loaded Successfully!</h4>
                    <p style="color: #94A3B8; margin: 0.5rem 0 0 0;">
                    <strong>File:</strong> {st.session_state.uploaded_filename}
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2rem; color: #10B981;">📊</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show quick stats
        render_section_header("📈 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", stats['total_students'])
        with col2:
            st.metric("Pass Rate", f"{stats['pass_rate']:.1f}%")
        with col3:
            st.metric("Avg Final Grade", f"{stats['avg_final_grade']:.1f}/20")
        with col4:
            st.metric("At Risk", stats['at_risk'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("High Achievers", stats['high_achievers'])
        with col2:
            st.metric("Avg Study Time", f"{stats['avg_studytime']:.1f}/4")
        with col3:
            st.metric("Avg Absences", f"{stats['avg_absences']:.1f}")
        with col4:
            st.metric("Perfect Attendance", stats['perfect_attendance'])
        
        # Navigation prompt
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h3 style="color: #38BDF8;">Ready to Analyze! 🚀</h3>
            <p style="color: #94A3B8; margin-bottom: 1.5rem;">Use the sidebar or click the cards below to navigate through the analysis pipeline:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create interactive navigation cards using Streamlit columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(
                label="📊\nData Overview\n\nExplore your data",
                use_container_width=True,
                help="Navigate to Data Overview page",
                key="home_nav_data"
            ):
                st.session_state.current_menu = "📊 Data Overview"
                st.rerun()
        
        with col2:
            if st.button(
                label="🎯\nClustering\n\nDiscover patterns",
                use_container_width=True,
                help="Navigate to Clustering Analysis page",
                key="home_nav_clustering"
            ):
                st.session_state.current_menu = "🎯 Clustering"
                st.rerun()
        
        with col3:
            if st.button(
                label="🤖\nML Model\n\nPredict performance",
                use_container_width=True,
                help="Navigate to Machine Learning Model page",
                key="home_nav_mlmodel"
            ):
                st.session_state.current_menu = "🤖 ML Model"
                st.rerun()
        
        
        with col4:
            if st.button(
                label="🔮\nPredict\n\nMake predictions",
                use_container_width=True,
                help="Navigate to Prediction page",
                key="home_nav_predict"
            ):
                st.session_state.current_menu = "🔮 Predict"
                st.rerun()
        
        # Add custom CSS styling for the navigation buttons
        st.markdown("""
        <style>
            /* Custom styling for the navigation card buttons */
            div[data-testid="column"] button {
                padding: 1.5rem !important;
                background: #1E293B !important;
                border-radius: 10px !important;
                border: 2px solid #334155 !important;
                color: #E2E8F0 !important;
                font-weight: 600 !important;
                font-size: 1rem !important;
                transition: all 0.3s !important;
                height: auto !important;
                min-height: 150px !important;
                white-space: pre-line !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: center !important;
                line-height: 1.5 !important;
            }
            
            div[data-testid="column"] button:hover {
                border-color: #3B82F6 !important;
                transform: translateY(-5px) !important;
                background: #1E293B !important;
            }
            
            div[data-testid="column"] button:active {
                border-color: #3B82F6 !important;
                background: #1E293B !important;
            }
            
            /* Style for button text */
            div[data-testid="column"] button p {
                margin: 0;
                text-align: center;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Additional tip
        st.markdown("""
        <div style="margin-top: 2rem;">
            <p style="color: #94A3B8; text-align: center; font-size: 0.9rem;">
                💡 <strong>Tip:</strong> You can also use the sidebar navigation for quick access to all sections
            </p>
        </div>
        """, unsafe_allow_html=True)

def show_data_overview():
    """Data overview page"""
    render_header("Data Overview", "Explore and Understand Your Data")
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="status-warning">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <div>
                    <h4 style="color: #F59E0B; margin: 0;">Data Not Loaded</h4>
                    <p style="color: #94A3B8; margin: 0.25rem 0 0 0;">
                    Please upload data from the Home page first.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    analyzer = st.session_state.analyzer
    viz = st.session_state.viz
    
    # Create a comprehensive overview with tabs
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 Dataset Summary", "🔍 Data Exploration", "📈 Statistical Analysis"])
    
    with main_tab1:
        # Dataset Overview Section
        render_section_header("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(analyzer.df))
        with col2:
            st.metric("Total Features", len(analyzer.df.columns))
        with col3:
            st.metric("Numeric Columns", len(analyzer.df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Categorical Columns", len(analyzer.df.select_dtypes(include=['object']).columns))
        
        # Memory Usage
        memory_usage = analyzer.df.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
        
        # Data Quality Section
        render_section_header("🧹 Data Quality Assessment")
        
        # Calculate missing values
        missing_values = analyzer.df.isnull().sum()
        total_missing = missing_values.sum()
        missing_percentage = (total_missing / (len(analyzer.df) * len(analyzer.df.columns))) * 100
        
        # Data types distribution
        dtype_counts = analyzer.df.dtypes.value_counts()
        
        # Create quality metrics in columns
        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        with q_col1:
            st.metric(
                "Missing Values", 
                total_missing,
                delta=f"{missing_percentage:.2f}% of total"
            )
        with q_col2:
            st.metric(
                "Memory Usage", 
                f"{memory_usage:.2f} MB"
            )
        with q_col3:
            duplicate_count = analyzer.df.duplicated().sum()
            st.metric(
                "Duplicates", 
                duplicate_count
            )
        with q_col4:
            st.metric(
                "Complete Cases", 
                analyzer.df.dropna().shape[0],
                delta=f"{analyzer.df.dropna().shape[0]}/{len(analyzer.df)} rows"
            )
        
        # Detailed Missing Values Analysis
        with st.expander("🔍 Missing Values Analysis", expanded=False):
            if total_missing > 0:
                # Create missing values dataframe
                missing_df = pd.DataFrame({
                    'Feature': missing_values.index,
                    'Missing Count': missing_values.values,
                    'Missing %': (missing_values.values / len(analyzer.df)) * 100
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Missing Values by Feature:**")
                    st.dataframe(missing_df, use_container_width=True)
                
                with col2:
                    st.markdown("**Missing Data Heatmap:**")
                    # Create a simple heatmap visualization
                    fig_missing = go.Figure(data=go.Heatmap(
                        z=analyzer.df.isnull().astype(int).head(50).values.T,
                        colorscale='Reds',
                        showscale=False
                    ))
                    fig_missing.update_layout(
                        title="First 50 Rows - Missing Data",
                        height=300,
                        xaxis_title="Row Index",
                        yaxis_title="Features"
                    )
                    st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("✅ No missing values found in the dataset!")
        
        # Data Types Analysis
        with st.expander("📊 Data Types Distribution", expanded=False):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Create bar chart of data types
                dtype_df = pd.DataFrame({
                    'Data Type': dtype_counts.index.astype(str),
                    'Count': dtype_counts.values
                })
                
                fig_dtype = px.bar(
                    dtype_df,
                    x='Data Type',
                    y='Count',
                    title='Data Type Distribution',
                    color='Count',
                    color_continuous_scale='Teal'
                )
                st.plotly_chart(fig_dtype, use_container_width=True)
            
            with col2:
                st.markdown("**Data Type Details:**")
                dtype_details = []
                for dtype, count in dtype_counts.items():
                    features = analyzer.df.select_dtypes(include=[dtype]).columns.tolist()
                    dtype_details.append({
                        'Type': str(dtype),
                        'Count': count,
                        'Features': features[:5]  # Show first 5 features
                    })
                
                for detail in dtype_details:
                    st.write(f"**{detail['Type']}** ({detail['Count']} columns)")
                    if detail['Features']:
                        st.write(f"Examples: {', '.join(detail['Features'])}")
                    if len(detail['Features']) > 5:
                        st.write(f"... and {len(detail['Features']) - 5} more")
                    st.write("---")
        
        # Basic Statistics Section
        render_section_header("📈 Basic Statistics")
        
        # Quick stats cards
        stats = analyzer.get_academic_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pass Rate", f"{stats['pass_rate']:.1f}%")
        with col2:
            st.metric("Avg Final Grade", f"{stats['avg_final_grade']:.1f}/20")
        with col3:
            st.metric("High Achievers", stats['high_achievers'])
        with col4:
            st.metric("At Risk", stats['at_risk'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Study Time", f"{stats['avg_studytime']:.1f}/4")
        with col2:
            st.metric("Avg Absences", f"{stats['avg_absences']:.1f}")
        with col3:
            st.metric("Perfect Attendance", stats['perfect_attendance'])
        with col4:
            st.metric("Total Students", stats['total_students'])
    
    with main_tab2:
        # Interactive Data Exploration
        render_section_header("🔍 Interactive Data Explorer")
        
        # Data Preview Controls
        st.markdown("### 📋 Data Preview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sample_size = st.slider("Sample Size", 10, 100, 20, help="Number of rows to display")
        with col2:
            sort_column = st.selectbox("Sort By", ["None"] + analyzer.df.columns.tolist())
        with col3:
            sort_order = st.selectbox("Order", ["Ascending", "Descending"]) if sort_column != "None" else "None"
        
        # Prepare sample data
        sample_df = analyzer.df.copy()
        
        if sort_column != "None":
            sample_df = sample_df.sort_values(
                by=sort_column, 
                ascending=(sort_order == "Ascending")
            )
        
        # Display sample data
        st.dataframe(sample_df.head(sample_size), use_container_width=True)
        
        # Display data shape
        st.caption(f"Showing {min(sample_size, len(sample_df))} of {len(sample_df)} rows and {len(sample_df.columns)} columns")
        
        # Column Analysis
        render_section_header("🎯 Column Analysis")
        
        selected_column = st.selectbox(
            "Select a column for detailed analysis:",
            analyzer.df.columns.tolist(),
            index=0 if 'G3' in analyzer.df.columns else 0
        )
        
        if selected_column:
            col_type = analyzer.df[selected_column].dtype
            unique_values = analyzer.df[selected_column].nunique()
            null_count = analyzer.df[selected_column].isnull().sum()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Type", str(col_type))
            with col2:
                st.metric("Unique Values", unique_values)
            with col3:
                st.metric("Null Values", null_count)
            with col4:
                st.metric(
                    "Null %", 
                    f"{(null_count / len(analyzer.df)) * 100:.2f}%"
                )
            
            # Value Distribution
            if pd.api.types.is_numeric_dtype(col_type):
                # For numeric columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Basic Statistics:**")
                    stats_df = analyzer.df[selected_column].describe().reset_index()
                    stats_df.columns = ['Statistic', 'Value']
                    st.dataframe(stats_df, use_container_width=True)
                
                with col2:
                    # Create histogram
                    fig_dist = px.histogram(
                        analyzer.df,
                        x=selected_column,
                        title=f'Distribution of {selected_column}',
                        nbins=30,
                        color_discrete_sequence=['#3B82F6'],
                        opacity=0.8
                    )
                    fig_dist.update_layout(height=400)
                    st.plotly_chart(fig_dist, use_container_width=True)
            else:
                # For categorical columns
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Value counts
                    value_counts = analyzer.df[selected_column].value_counts().head(10)
                    st.markdown("**Top 10 Values:**")
                    for value, count in value_counts.items():
                        percentage = (count / len(analyzer.df)) * 100
                        st.write(f"• {value}: {count} ({percentage:.1f}%)")
                
                with col2:
                    # Create bar chart
                    value_counts_df = value_counts.reset_index()
                    value_counts_df.columns = ['Value', 'Count']
                    
                    fig_bar = px.bar(
                        value_counts_df.head(10),
                        x='Value',
                        y='Count',
                        title=f'Top Values in {selected_column}',
                        color='Count',
                        color_continuous_scale='Teal'
                    )
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # Show sample values
            with st.expander("👀 Sample Values from this Column"):
                unique_sample = analyzer.df[selected_column].dropna().unique()[:20]
                st.write(", ".join([str(val) for val in unique_sample]))
    
    with main_tab3:
        # Statistical Analysis
        render_section_header("📊 Statistical Analysis")
        
        # Correlation Analysis
        st.markdown("### 🔗 Feature Correlation Matrix")
        correlation_matrix = analyzer.get_correlation_matrix()
        
        # Display full correlation matrix with options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Correlation matrix visualization
            fig_corr = viz.create_correlation_heatmap(correlation_matrix)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.markdown("**🔍 Correlation Explorer**")
            
            # Select features for specific correlation
            feature1 = st.selectbox(
                "First Feature:",
                correlation_matrix.columns.tolist(),
                index=correlation_matrix.columns.get_loc('G3') if 'G3' in correlation_matrix.columns else 0
            )
            
            feature2 = st.selectbox(
                "Second Feature:",
                correlation_matrix.columns.tolist(),
                index=correlation_matrix.columns.get_loc('G2') if 'G2' in correlation_matrix.columns else 1
            )
            
            if feature1 and feature2:
                correlation_value = correlation_matrix.loc[feature1, feature2]
                
                st.metric(
                    f"{feature1} ↔ {feature2}",
                    f"{correlation_value:.3f}",
                    delta="Strong" if abs(correlation_value) > 0.7 else 
                          "Moderate" if abs(correlation_value) > 0.3 else "Weak"
                )
                
                # Interpretation
                if correlation_value > 0.7:
                    st.success("Strong positive correlation")
                elif correlation_value > 0.3:
                    st.info("Moderate positive correlation")
                elif correlation_value > -0.3:
                    st.warning("Weak or no correlation")
                elif correlation_value > -0.7:
                    st.info("Moderate negative correlation")
                else:
                    st.error("Strong negative correlation")
        
        # Top Correlations
        render_section_header("🏆 Strongest Correlations")
        
        # Calculate top positive and negative correlations
        corr_series = correlation_matrix.unstack()
        corr_series = corr_series[corr_series < 1].sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Positive Correlations:**")
            top_positive = corr_series.head(10)
            for pair, value in top_positive.items():
                st.write(f"• `{pair[0]}` ↔ `{pair[1]}`: **{value:.3f}**")
        
        with col2:
            st.markdown("**Top Negative Correlations:**")
            top_negative = corr_series.tail(10)[::-1]
            for pair, value in top_negative.items():
                st.write(f"• `{pair[0]}` ↔ `{pair[1]}`: **{value:.3f}**")
        
        # Distribution Analysis
        render_section_header("📈 Feature Distributions")
        
        # Select features for distribution analysis
        features_to_plot = st.multiselect(
            "Select features to visualize:",
            analyzer.df.select_dtypes(include=[np.number]).columns.tolist(),
            default=['G3', 'G2', 'G1', 'studytime', 'absences'] if all(col in analyzer.df.columns for col in ['G3', 'G2', 'G1', 'studytime', 'absences']) else analyzer.df.select_dtypes(include=[np.number]).columns.tolist()[:5]
        )
        
        if features_to_plot:
            # Create subplots for selected features
            fig_distributions = viz.create_feature_distributions(analyzer.df[features_to_plot])
            st.plotly_chart(fig_distributions, use_container_width=True)
        
        # Outlier Detection - EXACTLY as in notebook
        render_section_header("⚠️ Outlier Detection")

        if st.checkbox("Show outlier analysis (notebook method)"):
            cols = ['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime',
                    'goout','Dalc','Walc','health','absences','G1','G2','G3']
            
            # Filter to only include columns that exist in the dataset
            cols = [col for col in cols if col in analyzer.df.columns]
            
            outlier_records = []
            
            for col in cols:
                Q1 = analyzer.df[col].quantile(0.25)
                Q3 = analyzer.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Create mask for outliers
                mask = (analyzer.df[col] < Q1 - 1.5 * IQR) | (analyzer.df[col] > Q3 + 1.5 * IQR)
                
                # Get outlier records
                for idx in analyzer.df[mask].index:
                    outlier_records.append({
                        'Row Index': idx,
                        'Feature': col,
                        'Value': analyzer.df.loc[idx, col],
                        'Lower Bound': f"{Q1 - 1.5 * IQR:.2f}",
                        'Upper Bound': f"{Q3 + 1.5 * IQR:.2f}"
                    })
            
            if outlier_records:
                outliers_df = pd.DataFrame(outlier_records)
                outliers_df = outliers_df.sort_values('Row Index').reset_index(drop=True)
                
                # Display results
                st.markdown(f"**Found {len(outliers_df)} outlier records**")
                
                # Show summary statistics
                summary = outliers_df['Feature'].value_counts().reset_index()
                summary.columns = ['Feature', 'Outlier Count']
                summary['Outlier %'] = (summary['Outlier Count'] / len(analyzer.df) * 100).round(2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Outlier Count by Feature:**")
                    st.dataframe(
                        summary,
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    # Create bar chart
                    fig = px.bar(
                        summary,
                        x='Feature',
                        y='Outlier Count',
                        title='Outlier Distribution by Feature',
                        color='Outlier Count',
                        color_continuous_scale='reds'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed outlier records
                with st.expander(f"🔍 View {len(outliers_df)} Detailed Outlier Records"):
                    st.dataframe(
                        outliers_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Row Index": st.column_config.NumberColumn("Row #"),
                            "Feature": st.column_config.TextColumn("Feature"),
                            "Value": st.column_config.NumberColumn("Actual Value", format="%.2f"),
                            "Lower Bound": st.column_config.TextColumn("Lower Bound"),
                            "Upper Bound": st.column_config.TextColumn("Upper Bound")
                        }
                    )
                    
                    # Add download button
                    csv = outliers_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Outlier Records",
                        data=csv,
                        file_name="outlier_records.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.success("✅ No outliers found in the selected features!")
        
        # Data Export Options
        render_section_header("💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Cleaned Dataset", use_container_width=True):
                csv = analyzer.df.to_csv(index=False)
                st.download_button(
                    label="Click to download",
                    data=csv,
                    file_name="student_data_cleaned.csv",
                    mime="text/csv",
                    key="download_cleaned"
                )
        
        with col2:
            if st.button("📊 Download Statistics", use_container_width=True):
                # Generate comprehensive statistics
                stats_summary = pd.DataFrame({
                    'Feature': analyzer.df.columns,
                    'Data Type': analyzer.df.dtypes.astype(str),
                    'Missing Values': analyzer.df.isnull().sum().values,
                    'Missing %': (analyzer.df.isnull().sum().values / len(analyzer.df)) * 100,
                    'Unique Values': [analyzer.df[col].nunique() for col in analyzer.df.columns]
                })
                
                csv_stats = stats_summary.to_csv(index=False)
                st.download_button(
                    label="Click to download",
                    data=csv_stats,
                    file_name="dataset_statistics.csv",
                    mime="text/csv",
                    key="download_stats"
                )
        
        with col3:
            if st.button("📋 Download Summary Report", use_container_width=True):
                # Create a comprehensive summary report
                report_lines = [
                    "DATASET SUMMARY REPORT",
                    "=" * 40,
                    f"Generated: {pd.Timestamp.now()}",
                    f"Total Records: {len(analyzer.df)}",
                    f"Total Features: {len(analyzer.df.columns)}",
                    f"Memory Usage: {memory_usage:.2f} MB",
                    "\nDATA QUALITY:",
                    f"Missing Values: {total_missing} ({missing_percentage:.2f}%)",
                    f"Duplicates: {duplicate_count}",
                    "\nDATA TYPES:",
                ]
                
                for dtype, count in dtype_counts.items():
                    report_lines.append(f"  {dtype}: {count} columns")
                
                report_lines.append("\nKEY STATISTICS:")
                for key, value in stats.items():
                    report_lines.append(f"  {key}: {value}")
                
                report_text = "\n".join(report_lines)
                
                st.download_button(
                    label="Click to download",
                    data=report_text,
                    file_name="dataset_summary_report.txt",
                    mime="text/plain",
                    key="download_report"
                )

def show_clustering_analysis():
    """Clustering analysis page"""
    render_header("Clustering Analysis", "Discover Student Behavioral Patterns")
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="status-warning">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <div>
                    <h4 style="color: #F59E0B; margin: 0;">Data Not Loaded</h4>
                    <p style="color: #94A3B8; margin: 0.25rem 0 0 0;">
                    Please upload data from the Home page first.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    analyzer = st.session_state.analyzer
    viz = st.session_state.viz
    
    # Clustering Control Section
    render_section_header("🎯 Run Clustering Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #38BDF8;">📊 Clustering Methodology (Exact Notebook Implementation)</h4>
            <p style="color: #CBD5E1; line-height: 1.6;">
            The clustering follows the exact steps from the notebook:
            </p>
            <ol style="color: #94A3B8;">
                <li>Exclude grades (G1, G2, G3) for pure behavioral clustering</li>
                <li>Standardize features using StandardScaler</li>
                <li>Use K-Means with k-means++ initialization</li>
                <li>Apply elbow method to determine optimal clusters</li>
                <li>Calculate silhouette score for validation</li>
                <li>Create cluster_weight feature based on average G3</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        n_clusters = st.slider("Number of Clusters", 2, 6, 2, key="clustering_slider")
        
        if st.button("🚀 Run Clustering", use_container_width=True, type="primary"):
            with st.spinner(f"Performing clustering with {n_clusters} clusters..."):
                results = analyzer.perform_clustering(n_clusters=n_clusters)
                st.session_state.clustering_done = True
                st.session_state.clustering_results = results
                st.success(f"✅ Clustering complete! Created {n_clusters} clusters.")
                st.rerun()
    
    if st.session_state.clustering_done:
        results = st.session_state.clustering_results
        
        # Performance Metrics
        render_section_header("📊 Clustering Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Silhouette Score", f"{results['silhouette_score']:.4f}")
        with col2:
            st.metric("Number of Clusters", results['n_clusters'])
        with col3:
            st.metric("WCSS", f"{results['wcss'][results['n_clusters']-1]:.1f}")
        
        # Create tabs for different visualization types
        render_section_header("📈 Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Elbow Method", "2D Scatter", "3D Scatter", "Cluster Insights"])
        
        with tab1:
            render_section_header("Elbow Method Analysis")
            fig_elbow = viz.create_elbow_method_plot(results['wcss'], selected_clusters=results['n_clusters'])
            st.plotly_chart(fig_elbow, use_container_width=True)
            
            # Elbow method explanation
            st.markdown("""
            <div class="info-card">
                <h4 style="color: #38BDF8;">📊 Understanding the Elbow Method</h4>
                <p style="color: #CBD5E1; line-height: 1.6;">
                The elbow method helps determine the optimal number of clusters by looking for the "elbow" point 
                where adding more clusters doesn't significantly reduce the Within-Cluster Sum of Squares (WCSS).
                </p>
                <p style="color: #94A3B8;">
                <strong>Interpretation:</strong> Look for the point where the rate of decrease in WCSS sharply changes.
                This typically indicates the optimal number of clusters for your data.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            render_section_header("Interactive 2D Scatter Plot")
            
            # Feature selection for 2D scatter
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("X-axis Feature", ['studytime', 'absences', 'failures', 'G1', 'G2', 'G3'], index=0)
            with col2:
                y_feature = st.selectbox("Y-axis Feature", ['studytime', 'absences', 'failures', 'G1', 'G2', 'G3'], index=5)
            
            fig_scatter = viz.create_cluster_scatter(analyzer.df, x_feature, y_feature)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Interpretation guidance
            st.markdown("""
            <div class="info-card">
                <h4 style="color: #38BDF8;">🔍 Interpreting 2D Scatter Plots</h4>
                <p style="color: #CBD5E1; line-height: 1.6;">
                This plot shows how students are distributed across different behavioral dimensions, 
                colored by their assigned clusters. Look for:
                </p>
                <ul style="color: #94A3B8;">
                    <li><strong>Cluster separation:</strong> Well-separated clusters indicate good clustering</li>
                    <li><strong>Cluster shape:</strong> Circular clusters suggest K-means is appropriate</li>
                    <li><strong>Cluster density:</strong> Dense clusters indicate similar behaviors</li>
                    <li><strong>Outliers:</strong> Points far from clusters may need special attention</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            render_section_header("Interactive 3D Cluster Visualization")
            
            # Feature selection for 3D scatter
            col1, col2, col3 = st.columns(3)
            with col1:
                x_feature_3d = st.selectbox("X-axis Feature", 
                                          ['studytime', 'absences', 'failures', 'G1', 'G2', 'G3'], 
                                          index=0, key="x_3d")
            with col2:
                y_feature_3d = st.selectbox("Y-axis Feature", 
                                          ['studytime', 'absences', 'failures', 'G1', 'G2', 'G3'], 
                                          index=1, key="y_3d")
            with col3:
                z_feature_3d = st.selectbox("Z-axis Feature", 
                                          ['studytime', 'absences', 'failures', 'G1', 'G2', 'G3'], 
                                          index=5, key="z_3d")
            
            fig_3d = viz.create_3d_cluster_scatter(analyzer.df, x_feature_3d, y_feature_3d, z_feature_3d)
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # 3D visualization tips
            st.markdown("""
            <div class="info-card">
                <h4 style="color: #38BDF8;">🎮 3D Visualization Controls</h4>
                <p style="color: #CBD5E1; line-height: 1.6;">
                Use the following controls to explore the 3D visualization:
                </p>
                <ul style="color: #94A3B8;">
                    <li><strong>Rotate:</strong> Click and drag to rotate the 3D plot</li>
                    <li><strong>Zoom:</strong> Use mouse wheel or pinch gesture</li>
                    <li><strong>Pan:</strong> Right-click and drag to pan</li>
                    <li><strong>Reset:</strong> Double-click to reset the view</li>
                    <li><strong>Hover:</strong> Hover over points to see detailed information</li>
                </ul>
                <p style="color: #CBD5E1;">
                <strong>Tip:</strong> Look for clusters that are well-separated in 3D space, which indicates 
                distinct behavioral patterns across multiple dimensions.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab4:
            # CLUSTER PROFILES SECTION (like in the old Insights tab)
            render_section_header("👥 Cluster Profiles")
            
            profiles = analyzer.get_cluster_profiles()
            
            if profiles:
                for cluster_name, profile in profiles.items():
                    # Determine card color based on performance
                    if profile['pass_rate'] >= 70:
                        border_color = "#10B981"  # Green
                        status_emoji = "✅"
                    elif profile['pass_rate'] >= 50:
                        border_color = "#F59E0B"  # Yellow
                        status_emoji = "⚠️"
                    else:
                        border_color = "#EF4444"  # Red
                        status_emoji = "❌"
                    
                    # Cluster Profile Card
                    st.markdown(f"""
                    <div style="background: linear-gradient(145deg, #1E293B, #0F172A); 
                                border-radius: 12px; 
                                padding: 1.5rem; 
                                margin: 1.5rem 0;
                                border-left: 5px solid {border_color};
                                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <h3 style="color: #60A5FA; margin: 0;">{cluster_name}</h3>
                            <span style="background: rgba{tuple(int(border_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}; 
                                       color: {border_color}; 
                                       padding: 0.25rem 0.75rem; 
                                       border-radius: 20px;
                                       font-weight: 600;">
                                {status_emoji} {profile['pass_rate']:.1f}% Pass Rate
                            </span>
                        </div>
                        <p style="color: #94A3B8; font-style: italic; margin-bottom: 1.5rem;">{profile['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics for this cluster
                    cols = st.columns(4)
                    metrics = [
                        ("Students", f"{profile['size']}", "#3B82F6"),
                        ("Avg Grade", f"{profile['avg_grade']:.1f}/20", "#8B5CF6"),
                        ("Study Time", f"{profile['avg_studytime']:.1f}/4", "#10B981"),
                        ("Absences", f"{profile['avg_absences']:.1f}", "#F59E0B")
                    ]
                    
                    for col, (label, value, color) in zip(cols, metrics):
                        with col:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 0.75rem; background: rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}; 
                                        border-radius: 8px; border: 1px solid {color + '40'};">
                                <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{value}</div>
                                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.25rem;">{label}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Recommendations
                    with st.expander(f"🎯 Recommendations for {cluster_name}"):
                        if profile['pass_rate'] < 50:
                            st.error("**High Priority Interventions Needed:**")
                            st.markdown("""
                            - **Immediate academic support** programs
                            - **Attendance monitoring** and follow-up
                            - **Study skills workshops** and tutoring
                            - **Parent-teacher conferences** for intervention planning
                            - **Weekly progress tracking** and feedback
                            """)
                        elif profile['pass_rate'] < 70:
                            st.warning("**Moderate Support Recommended:**")
                            st.markdown("""
                            - **Additional tutoring** sessions
                            - **Study group** facilitation
                            - **Time management** training
                            - **Regular check-ins** with counselors
                            - **Goal setting** and progress monitoring
                            """)
                        else:
                            st.success("**Enhancement Opportunities:**")
                            st.markdown("""
                            - **Advanced coursework** options
                            - **Leadership development** programs
                            - **Peer mentoring** opportunities
                            - **Research projects** and internships
                            - **Honors program** enrollment
                            """)
                    
                    st.markdown("---")
            
            # Cluster Statistics
            render_section_header("📊 Cluster Statistics")
            cluster_stats = analyzer.get_cluster_statistics()
            if cluster_stats is not None:
                st.dataframe(cluster_stats, use_container_width=True)
                
                # Download button
                csv = cluster_stats.to_csv(index=False)
                st.download_button(
                    label="📥 Download Cluster Statistics",
                    data=csv,
                    file_name="cluster_statistics.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Feature Distribution Analysis
            render_section_header("📈 Feature Distribution Analysis")
            feature_to_compare = st.selectbox(
                "Select feature to compare across clusters:",
                ['G3', 'studytime', 'absences', 'failures', 'G1', 'G2', 'cluster_weight'],
                index=0,
                key="cluster_insights_feature"
            )
            fig_box = viz.create_box_plot_comparison(analyzer.df, feature_to_compare)
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Cluster Composition
            render_section_header("👥 Cluster Composition")
            composition = analyzer.get_cluster_composition()
            if composition:
                fig_composition = viz.create_cluster_composition_chart(composition)
                st.plotly_chart(fig_composition, use_container_width=True)
            
            # Cluster Comparison Radar Chart
            render_section_header("📈 Cluster Comparison")
            
            if profiles:
                fig_radar = viz.create_cluster_radar_chart(profiles)
                if fig_radar:
                    st.plotly_chart(fig_radar, use_container_width=True, height=600)

            
            # Key Insights
            render_section_header("💡 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="info-card">
                    <h4 style="color: #38BDF8;">Academic Insights</h4>
                    <ul style="color: #CBD5E1;">
                        <li>Study time consistency is the strongest predictor of success</li>
                        <li>Previous academic failures significantly impact current performance</li>
                        <li>Regular attendance shows strong correlation with higher grades</li>
                        <li>Mid-term grades (G2) are excellent predictors of final performance</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="info-card">
                    <h4 style="color: #38BDF8;">Actionable Recommendations</h4>
                    <ul style="color: #CBD5E1;">
                        <li>Implement targeted interventions based on cluster profiles</li>
                        <li>Use mid-term grades for early identification of at-risk students</li>
                        <li>Develop cluster-specific support programs</li>
                        <li>Monitor attendance patterns and intervene early</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Download Section
            render_section_header("📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = analyzer.df.to_csv(index=False)
                st.download_button(
                    label="📄 Full Dataset",
                    data=csv_data,
                    file_name="student_analysis_full.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="export_full_cluster"
                )
            
            with col2:
                if cluster_stats is not None:
                    csv_cluster = cluster_stats.to_csv(index=False)
                    st.download_button(
                        label="📊 Cluster Statistics",
                        data=csv_cluster,
                        file_name="cluster_statistics.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="export_cluster_stats"
                    )
            
            with col3:
                # Export cluster profiles as JSON
                if profiles:
                    import json
                    profiles_json = json.dumps(profiles, indent=2)
                    st.download_button(
                        label="🤖 Cluster Profiles",
                        data=profiles_json,
                        file_name="cluster_profiles.json",
                        mime="application/json",
                        use_container_width=True,
                        key="export_cluster_profiles"
                    )

def show_hybrid_model():
    """Machine Learning model page"""
    render_header("Machine Learning Model", "Hybrid RandomForest Prediction")
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="status-warning">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <div>
                    <h4 style="color: #F59E0B; margin: 0;">Data Not Loaded</h4>
                    <p style="color: #94A3B8; margin: 0.25rem 0 0 0;">
                    Please upload data from the Home page first.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if not st.session_state.clustering_done:
        st.markdown("""
        <div class="status-warning">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <div>
                    <h4 style="color: #F59E0B; margin: 0;">Clustering Not Performed</h4>
                    <p style="color: #94A3B8; margin: 0.25rem 0 0 0;">
                    Please run clustering analysis first.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    analyzer = st.session_state.analyzer
    viz = st.session_state.viz
    
    # Model Training Section
    render_section_header("🤖 Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #38BDF8;">🧠 Hybrid Model Architecture (Exact Notebook Implementation)</h4>
            <p style="color: #CBD5E1; line-height: 1.6;">
            The hybrid model follows the exact methodology from the notebook:
            </p>
            <ol style="color: #94A3B8;">
                <li>Combine original features with clustering results</li>
                <li>Use RandomForest classifier with GridSearchCV</li>
                <li>Train on 80% of data, test on 20% with stratification</li>
                <li>Include cluster_id and cluster_weight features</li>
                <li>Use same hyperparameter grid as notebook</li>
                <li>Compare traditional vs hybrid model performance</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🚀 Train Hybrid Model", use_container_width=True, type="primary"):
            with st.spinner("Training hybrid model with GridSearchCV..."):
                model_results = analyzer.train_hybrid_model()
                st.session_state.model_trained = True
                st.session_state.model_results = model_results
                st.success("✅ Model training complete!")
                st.rerun()
    
    if st.session_state.model_trained:
        model_results = st.session_state.model_results
        
        # Model Performance
        render_section_header("📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{model_results['accuracy']:.4f}")
        with col2:
            best_params = model_results['best_params']
            st.metric("Estimators", best_params.get('n_estimators', 'N/A'))
        with col3:
            st.metric("Max Depth", str(best_params.get('max_depth', 'None')))
        with col4:
            st.metric("Min Split", best_params.get('min_samples_split', 'N/A'))
        
        # Model Comparison
        render_section_header("⚖️ Model Comparison")
        
        if st.button("🔄 Compare Traditional vs Hybrid", use_container_width=True):
            with st.spinner("Running model comparison..."):
                comparison_results = analyzer.compare_models()
                st.session_state.model_comparison_done = True
                st.session_state.comparison_results = comparison_results
                st.success("✅ Comparison complete!")
                st.rerun()
        
        if st.session_state.model_comparison_done:
            comparison_results = st.session_state.comparison_results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Traditional Model", 
                    f"{comparison_results['accuracy_without_cluster']:.4f}",
                    delta="Baseline"
                )
            with col2:
                improvement = comparison_results['improvement_percent']
                st.metric(
                    "Hybrid Model", 
                    f"{comparison_results['accuracy_with_cluster']:.4f}",
                    delta=f"{improvement:+.2f}%"
                )
            with col3:
                if improvement > 0:
                    st.success(f"✅ Improvement: {improvement:.2f}%")
                else:
                    st.warning(f"⚠️ No improvement: {improvement:.2f}%")
            
            # Comparison Chart
            fig_comparison = viz.create_model_comparison_chart(comparison_results)
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Detailed Analysis Tabs
        render_section_header("🔍 Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Classification Report", "Confusion Matrix"])
        
        with tab1:
            render_section_header("Top Feature Importances")
            fig_importance = viz.create_feature_importance_chart(model_results['feature_importances'])
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with tab2:
            render_section_header("Classification Report")
            fig_report = viz.create_classification_report_table(model_results['classification_report'])
            st.plotly_chart(fig_report, use_container_width=True)
        
        with tab3:
            render_section_header("Confusion Matrix")
            fig_cm = viz.create_confusion_matrix_heatmap(
                model_results['y_true'], 
                model_results['y_pred']
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # Model Management
        render_section_header("💾 Model Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Model", use_container_width=True):
                if analyzer.save_model():
                    st.success("✅ Model saved successfully!")
                else:
                    st.warning("⚠️ No model to save")
        
        with col2:
            if st.button("📤 Load Model", use_container_width=True):
                try:
                    analyzer.load_model()
                    st.success("✅ Model loaded successfully!")
                except:
                    st.error("❌ No saved model found")

def show_cluster_insights():
    """Cluster insights page"""
    render_header("Cluster Insights", "Detailed Profiles and Recommendations")
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="status-warning">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <div>
                    <h4 style="color: #F59E0B; margin: 0;">Data Not Loaded</h4>
                    <p style="color: #94A3B8; margin: 0.25rem 0 0 0;">
                    Please upload data from the Home page first.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if not st.session_state.clustering_done:
        st.markdown("""
        <div class="status-warning">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <div>
                    <h4 style="color: #F59E0B; margin: 0;">Clustering Not Performed</h4>
                    <p style="color: #94A3B8; margin: 0.25rem 0 0 0;">
                    Please run clustering analysis first from the Clustering page.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    analyzer = st.session_state.analyzer
    viz = st.session_state.viz
    
    # Cluster Profiles
    render_section_header("👥 Cluster Profiles")
    
    profiles = analyzer.get_cluster_profiles()
    
    if profiles:
        for cluster_name, profile in profiles.items():
            # Determine card color based on performance
            if profile['pass_rate'] >= 70:
                border_color = "#10B981"  # Green
                status_emoji = "✅"
            elif profile['pass_rate'] >= 50:
                border_color = "#F59E0B"  # Yellow
                status_emoji = "⚠️"
            else:
                border_color = "#EF4444"  # Red
                status_emoji = "❌"
            
            # Cluster Profile Card
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #1E293B, #0F172A); 
                        border-radius: 12px; 
                        padding: 1.5rem; 
                        margin: 1.5rem 0;
                        border-left: 5px solid {border_color};
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="color: #60A5FA; margin: 0;">{cluster_name}</h3>
                    <span style="background: rgba{tuple(int(border_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}; 
                               color: {border_color}; 
                               padding: 0.25rem 0.75rem; 
                               border-radius: 20px;
                               font-weight: 600;">
                        {status_emoji} {profile['pass_rate']:.1f}% Pass Rate
                    </span>
                </div>
                <p style="color: #94A3B8; font-style: italic; margin-bottom: 1.5rem;">{profile['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics for this cluster
            cols = st.columns(4)
            metrics = [
                ("Students", f"{profile['size']}", "#3B82F6"),
                ("Avg Grade", f"{profile['avg_grade']:.1f}/20", "#8B5CF6"),
                ("Study Time", f"{profile['avg_studytime']:.1f}/4", "#10B981"),
                ("Absences", f"{profile['avg_absences']:.1f}", "#F59E0B")
            ]
            
            for col, (label, value, color) in zip(cols, metrics):
                with col:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 0.75rem; background: rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}; 
                                border-radius: 8px; border: 1px solid {color + '40'};">
                        <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{value}</div>
                        <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.25rem;">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommendations
            with st.expander(f"🎯 Recommendations for {cluster_name}"):
                if profile['pass_rate'] < 50:
                    st.error("**High Priority Interventions Needed:**")
                    st.markdown("""
                    - **Immediate academic support** programs
                    - **Attendance monitoring** and follow-up
                    - **Study skills workshops** and tutoring
                    - **Parent-teacher conferences** for intervention planning
                    - **Weekly progress tracking** and feedback
                    """)
                elif profile['pass_rate'] < 70:
                    st.warning("**Moderate Support Recommended:**")
                    st.markdown("""
                    - **Additional tutoring** sessions
                    - **Study group** facilitation
                    - **Time management** training
                    - **Regular check-ins** with counselors
                    - **Goal setting** and progress monitoring
                    """)
                else:
                    st.success("**Enhancement Opportunities:**")
                    st.markdown("""
                    - **Advanced coursework** options
                    - **Leadership development** programs
                    - **Peer mentoring** opportunities
                    - **Research projects** and internships
                    - **Honors program** enrollment
                    """)
            
            st.markdown("---")
    
    # Cluster Composition Visualization
    render_section_header("📊 Cluster Composition")
    
    composition = analyzer.get_cluster_composition()
    if composition:
        fig_composition = viz.create_cluster_composition_chart(composition)
        st.plotly_chart(fig_composition, use_container_width=True)
    
    # Radar Chart Comparison
    render_section_header("📈 Cluster Comparison")
    
    if profiles:
        fig_radar = viz.create_cluster_radar_chart(profiles)
        if fig_radar:
            st.plotly_chart(fig_radar, use_container_width=True, height=600)

    
    # Key Insights
    render_section_header("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #38BDF8;">Academic Insights</h4>
            <ul style="color: #CBD5E1;">
                <li>Study time consistency is the strongest predictor of success</li>
                <li>Previous academic failures significantly impact current performance</li>
                <li>Regular attendance shows strong correlation with higher grades</li>
                <li>Mid-term grades (G2) are excellent predictors of final performance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #38BDF8;">Actionable Recommendations</h4>
            <ul style="color: #CBD5E1;">
                <li>Implement targeted interventions based on cluster profiles</li>
                <li>Use mid-term grades for early identification of at-risk students</li>
                <li>Develop cluster-specific support programs</li>
                <li>Monitor attendance patterns and intervene early</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Download Section
    render_section_header("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = analyzer.df.to_csv(index=False)
        st.download_button(
            label="📄 Full Dataset",
            data=csv_data,
            file_name="student_analysis_full.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.session_state.clustering_done:
            cluster_stats = analyzer.get_cluster_statistics()
            if cluster_stats is not None:
                csv_cluster = cluster_stats.to_csv(index=False)
                st.download_button(
                    label="📊 Cluster Statistics",
                    data=csv_cluster,
                    file_name="cluster_statistics.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    with col3:
        if st.session_state.model_trained:
            st.download_button(
                label="🤖 Model Report",
                data="Model performance report generated from analysis",
                file_name="model_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )

def show_prediction():
    """Prediction page for individual and batch predictions"""
    render_header("Performance Prediction", "Predict Student Success Using Trained Model")
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="status-warning">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <div>
                    <h4 style="color: #F59E0B; margin: 0;">Data Not Loaded</h4>
                    <p style="color: #94A3B8; margin: 0.25rem 0 0 0;">
                    Please upload data from the Home page first.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if not st.session_state.model_trained:
        st.markdown("""
        <div class="status-warning">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <div>
                    <h4 style="color: #F59E0B; margin: 0;">Model Not Trained</h4>
                    <p style="color: #94A3B8; margin: 0.25rem 0 0 0;">
                    Please train the model first from the ML Model page.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    analyzer = st.session_state.analyzer
    viz = st.session_state.viz
    
    # Prediction Mode Selection
    render_section_header("🎯 Prediction Mode")
    
    mode = st.radio(
        "Select prediction mode:",
        ["Single Student Prediction", "Batch Prediction"],
        horizontal=True,
        key="prediction_mode_radio"
    )
    
    if mode == "Single Student Prediction":
        show_single_prediction(analyzer, viz)
    else:
        show_batch_prediction(analyzer, viz)

def show_single_prediction(analyzer, viz):
    """Show single student prediction interface"""
    render_section_header("👤 Single Student Prediction")
    
    # Get model features for guidance
    if analyzer.rf_model is None:
        st.error("Model not available. Please train the model first.")
        return
    
    # Get required features
    required_features = analyzer.rf_model.feature_names_in_
    
    # Create input form
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #38BDF8;">📝 Input Student Features</h4>
        <p style="color: #CBD5E1; line-height: 1.6;">
        Enter the student's academic and behavioral data. The model will predict whether 
        the student is likely to pass (G3 ≥ 10) or fail.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create input form in columns
    col1, col2, col3 = st.columns(3)
    
    student_data = {}
    
    with col1:
        st.subheader("📊 Academic Performance")
        student_data['G2'] = st.slider("Mid-term Grade (G2)", 0, 20, 10, 
                                      help="Second period grade (0-20)")
        student_data['G1'] = st.slider("First Period Grade (G1)", 0, 20, 10,
                                      help="First period grade (0-20)")
        student_data['studytime'] = st.slider("Weekly Study Time", 1, 4, 2,
                                            help="Weekly study time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
    
    with col2:
        st.subheader("📈 Behavioral Factors")
        student_data['failures'] = st.slider("Past Class Failures", 0, 4, 0,
                                           help="Number of past class failures (0-4)")
        student_data['absences'] = st.slider("Number of Absences", 0, 50, 5,
                                           help="Number of school absences (0-50)")
        student_data['age'] = st.slider("Age", 15, 22, 17,
                                      help="Student age (15-22)")
    
    with col3:
        st.subheader("🎯 Clustering Features")
        if 'cluster_id' in required_features:
            student_data['cluster_id'] = st.slider("Behavioral Cluster", 0, 5, 0,
                                                 help="Assigned behavioral cluster (0-5)")
        if 'cluster_weight' in required_features:
            student_data['cluster_weight'] = st.slider("Cluster Weight", 0.0, 20.0, 10.0, 0.1,
                                                     help="Cluster performance impact")
        
        # Add optional features
        st.subheader("⚙️ Additional Features")
        if 'Medu' in required_features:
            student_data['Medu'] = st.slider("Mother's Education", 0, 4, 2,
                                           help="Mother's education level (0-4)")
    
    # Add missing features with default values
    for feature in required_features:
        if feature not in student_data:
            if feature in analyzer.df.columns:
                student_data[feature] = analyzer.df[feature].median()
            else:
                student_data[feature] = 0
    
    # Prediction Button
    if st.button("🔮 Predict Student Performance", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            try:
                # Make prediction
                prediction = analyzer.predict_student_performance(student_data)
                
                # Display results
                st.markdown("---")
                render_section_header("📊 Prediction Results")
                
                # Result card
                if prediction['predicted_class'] == 'Pass':
                    card_color = "#10B981"
                    emoji = "✅"
                    message = "This student is predicted to PASS"
                else:
                    card_color = "#EF4444"
                    emoji = "❌"
                    message = "This student is predicted to FAIL"
                
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #1E293B, #0F172A); 
                            border-radius: 12px; 
                            padding: 2rem; 
                            margin: 1.5rem 0;
                            border-left: 5px solid {card_color};
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem;">
                        <h2 style="color: {card_color}; margin: 0;">{emoji} {prediction['predicted_class']}</h2>
                        <div style="font-size: 2.5rem; font-weight: 700; color: {card_color};">
                            {prediction['probability']*100:.1f}%
                        </div>
                    </div>
                    <p style="color: #CBD5E1; font-size: 1.1rem; margin-bottom: 0.5rem;">{message}</p>
                    <div style="display: flex; gap: 2rem; margin-top: 1.5rem;">
                        <div style="flex: 1;">
                            <div style="color: #94A3B8; font-size: 0.9rem;">Confidence</div>
                            <div style="color: #E2E8F0; font-size: 1.2rem; font-weight: 600;">
                                {prediction['confidence']*100:.1f}%
                            </div>
                        </div>
                        <div style="flex: 1;">
                            <div style="color: #94A3B8; font-size: 0.9rem;">Prediction Code</div>
                            <div style="color: #E2E8F0; font-size: 1.2rem; font-weight: 600;">
                                {prediction['prediction']} (1=Pass, 0=Fail)
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Get explanation
                explanation = analyzer.get_prediction_explanation(student_data)
                
                # Display key factors
                render_section_header("🔍 Key Factors Influencing Prediction")
                
                cols = st.columns(3)
                for idx, factor in enumerate(explanation['top_factors'][:3]):
                    with cols[idx]:
                        if factor['impact'] == 'Positive':
                            color = "#10B981"
                            icon = "📈"
                        elif factor['impact'] == 'Negative':
                            color = "#EF4444"
                            icon = "📉"
                        else:
                            color = "#F59E0B"
                            icon = "📊"
                        
                        st.markdown(f"""
                        <div style="background: rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}; 
                                    border-radius: 8px; 
                                    padding: 1rem; 
                                    border-left: 4px solid {color};">
                            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                                <span style="font-size: 1.2rem;">{icon}</span>
                                <div style="font-weight: 600; color: {color};">
                                    {factor['feature']}
                                </div>
                            </div>
                            <div style="color: #E2E8F0; font-size: 1.5rem; font-weight: 700;">
                                {factor['value']}
                            </div>
                            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.25rem;">
                                Impact: {factor['impact']}
                            </div>
                            <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 0.5rem;">
                                Importance: {factor['importance']:.4f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Get recommendations
                recommendations = analyzer.generate_recommendations(student_data)
                
                # Display recommendations
                render_section_header("🎯 Personalized Recommendations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.7); 
                                border-radius: 8px; 
                                padding: 1rem;
                                border-left: 4px solid #3B82F6;">
                        <h4 style="color: #60A5FA; margin-top: 0;">📋 Risk Assessment</h4>
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                            <span style="font-size: 1.5rem;">{"⚠️" if recommendations['risk_level'] == 'High' else "📊" if recommendations['risk_level'] == 'Moderate' else "✅"}</span>
                            <div>
                                <div style="color: #E2E8F0; font-weight: 600;">Risk Level: {recommendations['risk_level']}</div>
                                <div style="color: #94A3B8; font-size: 0.9rem;">
                                    Based on {prediction['probability']*100:.1f}% pass probability
                                </div>
                            </div>
                        </div>
                        <div style="color: #94A3B8;">
                            <strong>Key Areas:</strong> {', '.join(recommendations['key_areas']) if recommendations['key_areas'] else 'No specific areas identified'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.7); 
                                border-radius: 8px; 
                                padding: 1rem;
                                border-left: 4px solid #10B981;">
                        <h4 style="color: #60A5FA; margin-top: 0;">🚀 Action Items</h4>
                        <ul style="color: #CBD5E1; padding-left: 1.2rem;">
                            {''.join([f'<li>{item}</li>' for item in recommendations['action_items']]) if recommendations['action_items'] else '<li>No specific actions needed</li>'}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

def show_batch_prediction(analyzer, viz):
    """Show batch prediction interface"""
    render_section_header("👥 Batch Student Prediction")
    
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #38BDF8;">📁 Upload Student Data for Batch Prediction</h4>
        <p style="color: #CBD5E1; line-height: 1.6;">
        Upload a CSV file containing student data for batch prediction. The file should contain 
        all the required features used during model training.
        </p>
        <p style="color: #94A3B8; font-weight: 600;">Required Features:</p>
        <ul style="color: #94A3B8;">
            <li>G2: Mid-term grade (0-20)</li>
            <li>G1: First period grade (0-20)</li>
            <li>Other behavioral features from training</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload for batch prediction
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=['csv'],
        help="Upload student data CSV file with required features",
        key="batch_prediction_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            content = uploaded_file.getvalue().decode('utf-8')
            try:
                batch_df = pd.read_csv(io.StringIO(content), delimiter=';')
            except:
                batch_df = pd.read_csv(io.StringIO(content), delimiter=',')
            
            # Show preview
            st.markdown(f"**File Preview:** {len(batch_df)} students")
            st.dataframe(batch_df.head(10), use_container_width=True)
            
            if st.button("🔮 Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(batch_df)} students..."):
                    try:
                        # Make batch predictions
                        results = analyzer.batch_predict(batch_df)
                        
                        # Display statistics
                        render_section_header("📊 Batch Prediction Results")
                        
                        stats = results['statistics']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Students", stats['total_students'])
                        with col2:
                            st.metric("Predicted to Pass", stats['predicted_pass'])
                        with col3:
                            st.metric("Predicted to Fail", stats['predicted_fail'])
                        with col4:
                            st.metric("Pass Rate", f"{stats['pass_rate']:.1f}%")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("High Risk", stats['high_risk_students'])
                        with col2:
                            st.metric("Moderate Risk", stats['moderate_risk_students'])
                        with col3:
                            st.metric("Low Risk", stats['low_risk_students'])
                        
                        # Display predictions table
                        render_section_header("📋 Detailed Predictions")
                        st.dataframe(results['predictions'], use_container_width=True)
                        
                        # Download button for results
                        csv = results['predictions'].to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="student_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Visualization
                        render_section_header("📈 Prediction Distribution")
                        
                        # Create distribution chart
                        fig = viz.create_grade_distribution(results['predictions'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Batch prediction failed: {str(e)}")
                        st.info("Make sure your CSV file contains all required features.")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; margin-bottom: 1.5rem;">
            <h2 style="
                color: #60A5FA;
                font-size: 1.6rem;
                font-weight: 700;
                letter-spacing: 3px;
                line-height: 1.3;
            ">
                <span style="font-size: 3rem;">📚</span><br>
                Student Analytics
            </h2>
            <p style="color:#94A3B8; font-size:0.85rem;">
                Academic Performance Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

        
        # Navigation
        st.markdown("### 🧭 Navigation")
        
        # Create custom navigation buttons (removed "📈 Insights" from here)
        nav_options = ["🏠 Home", "📊 Data Overview", "🎯 Clustering", "🤖 ML Model", "🔮 Predict"]
        
        # Display each navigation option as a button
        for option in nav_options:
            is_active = st.session_state.current_menu == option
            button_class = "sidebar-button active" if is_active else "sidebar-button"
            
            if st.button(
                option,
                key=f"nav_{option}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_menu = option
                st.rerun()
        
        st.markdown("---")
        
        # Data Source Section
        st.markdown("### 📁 Data Source")
        
        # Create styled file uploader
        uploaded_file = st.file_uploader(
            "**Drag and drop a CSV file**\n\nOr click to browse\n\nSupports: .csv files",
            type=['csv'],
            help="Upload student performance data CSV file",
            label_visibility="collapsed",
            key="sidebar_file_uploader"
        )
        
        # Handle file upload
        if uploaded_file is not None and uploaded_file != st.session_state.get('last_uploaded_file'):
            if handle_file_upload(uploaded_file):
                st.session_state.last_uploaded_file = uploaded_file
                st.success(f"✅ {uploaded_file.name} uploaded!")
                st.rerun()
        
        st.markdown("---")
        
        # Current Data Info
        if st.session_state.data_loaded:
            st.markdown("### 📋 Current Data")
            st.write(f"📁 {st.session_state.uploaded_filename}")
            if st.session_state.analyzer:
                st.write(f"📊 {len(st.session_state.analyzer.df)} records")
                if 'cluster_id' in st.session_state.analyzer.df.columns:
                    unique_clusters = st.session_state.analyzer.df['cluster_id'].nunique()
                    st.write(f"🎯 {unique_clusters} clusters")
        
        # Reset Button
        if st.button("🔄 Reset Analysis", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                if key != 'viz':  # Keep visualizations
                    del st.session_state[key]
            init_session_state()
            st.success("Analysis reset successfully!")
            st.rerun()
    
    # Main content routing
    current_menu = st.session_state.current_menu
    
    if current_menu == "🏠 Home":
        show_home()
    elif current_menu == "📊 Data Overview":
        show_data_overview()
    elif current_menu == "🎯 Clustering":
        show_clustering_analysis()
    elif current_menu == "🤖 ML Model":
        show_hybrid_model()
    elif current_menu == "🔮 Predict":
        show_prediction()

if __name__ == "__main__":
    main()