import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# Set Streamlit page configuration
st.set_page_config(
    page_title="Census Income Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: #F8FAFC;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-title {
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 8px;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        margin: 0;
    }
    
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 18px 24px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94A3B8;
        font-weight: 500;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    
    .prediction-box-high {
        background: linear-gradient(135deg, #065F46 0%, #047857 100%);
        color: #ECFDF5;
        padding: 24px;
        border-radius: 14px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 20px rgba(4, 120, 87, 0.2);
    }
    
    .prediction-box-low {
        background: linear-gradient(135deg, #991B1B 0%, #B91C1C 100%);
        color: #FEF2F2;
        padding: 24px;
        border-radius: 14px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 20px rgba(185, 28, 28, 0.2);
    }
    
    .prediction-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    
    .prediction-desc {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0px 24px;
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

DATASET_PATH = "census-income.csv"

def get_safe_index(options, target_item, default_index=0):
    options_list = list(options)
    if target_item in options_list:
        return options_list.index(target_item)
    return min(default_index, max(0, len(options_list) - 1)) if options_list else 0

@st.cache_data
def load_data():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset file '{DATASET_PATH}' not found in root folder.")
        st.stop()
    df = pd.read_csv(DATASET_PATH)
    # Strip whitespace from string entries
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
    return df

@st.cache_resource
def train_model(df, max_depth, test_size, criterion):
    data_encoded = df.copy()
    encoders = {}
    
    for col in data_encoded.columns:
        if not pd.api.types.is_numeric_dtype(data_encoded[col]):
            le = LabelEncoder()
            data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
            encoders[col] = le
            
    X = data_encoded.drop('annual_income', axis=1)
    y = data_encoded['annual_income']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=34
    )
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=34
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=encoders['annual_income'].classes_, output_dict=True)
    
    return model, encoders, X_train, X_test, y_train, y_test, y_pred, acc, cm, cr

# Load data
df = load_data()

# Sidebar - Model Configuration
st.sidebar.markdown("### ⚙️ Model Parameters")
max_depth = st.sidebar.slider("Decision Tree Max Depth", min_value=1, max_value=20, value=7, step=1)
test_size = st.sidebar.slider("Test Split Ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
criterion = st.sidebar.selectbox("Split Criterion", options=["gini", "entropy", "log_loss"], index=0)

model, encoders, X_train, X_test, y_train, y_test, y_pred, acc, cm, cr = train_model(
    df, max_depth, test_size, criterion
)

# Header Section
st.markdown("""
    <div class="main-header">
        <h1 class="main-title">Census Income Predictor 💼</h1>
        <p class="main-subtitle">Classify individual annual income (&le;$50K vs &gt;$50K) using a Decision Tree Machine Learning Classifier based on Census Demographic Data.</p>
    </div>
""", unsafe_allow_html=True)

# Overview Metrics Cards
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Features</div>
            <div class="metric-value">{X_train.shape[1]}</div>
        </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">{acc * 100:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)
with m4:
    high_income_pct = (df['annual_income'] == '>50K').mean() * 100
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">&gt;50K Proportion</div>
            <div class="metric-value">{high_income_pct:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Income Predictor", "📊 Exploratory Data Analysis", "🌲 Decision Tree Explorer"])

# TAB 1: INCOME PREDICTOR
with tab1:
    st.subheader("Interactive Individual Feature Inputs")
    st.caption("Provide demographic and employment details below to run real-time Decision Tree inference.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", min_value=17, max_value=90, value=38)
        workclass = st.selectbox("Workclass", options=encoders['workclass'].classes_, index=get_safe_index(encoders['workclass'].classes_, 'Private', 4))
        fnlwgt = st.number_input("Fnlwgt (Final Weight)", min_value=10000, max_value=1500000, value=180000, step=10000)
        education = st.selectbox("Education Level", options=encoders['education'].classes_, index=get_safe_index(encoders['education'].classes_, 'Bachelors', 9))
        education_num = st.slider("Education Num (Years)", min_value=1, max_value=16, value=13)
        
    with col2:
        marital_status = st.selectbox("Marital Status", options=encoders['marital-status'].classes_, index=get_safe_index(encoders['marital-status'].classes_, 'Married-civ-spouse', 2))
        occupation = st.selectbox("Occupation", options=encoders['occupation'].classes_, index=get_safe_index(encoders['occupation'].classes_, 'Prof-specialty', 4))
        relationship = st.selectbox("Relationship", options=encoders['relationship'].classes_, index=get_safe_index(encoders['relationship'].classes_, 'Husband', 0))
        race = st.selectbox("Race", options=encoders['race'].classes_, index=get_safe_index(encoders['race'].classes_, 'White', 4))
        sex = st.radio("Sex", options=encoders['sex'].classes_, index=get_safe_index(encoders['sex'].classes_, 'Male', 1), horizontal=True)
        
    with col3:
        capital_gain = st.number_input("Capital Gain ($)", min_value=0, max_value=100000, value=0, step=500)
        capital_loss = st.number_input("Capital Loss ($)", min_value=0, max_value=5000, value=0, step=100)
        hours_per_week = st.slider("Hours per Week", min_value=1, max_value=99, value=40)
        native_country = st.selectbox("Native Country", options=encoders['native-country'].classes_, index=get_safe_index(encoders['native-country'].classes_, 'United-States', 39))

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Run Income Prediction", type="primary", use_container_width=True):
        # Prepare feature vector
        input_data = {
            'age': age,
            'workclass': encoders['workclass'].transform([workclass])[0],
            'fnlwgt': fnlwgt,
            'education': encoders['education'].transform([education])[0],
            'education-num': education_num,
            'marital-status': encoders['marital-status'].transform([marital_status])[0],
            'occupation': encoders['occupation'].transform([occupation])[0],
            'relationship': encoders['relationship'].transform([relationship])[0],
            'race': encoders['race'].transform([race])[0],
            'sex': encoders['sex'].transform([sex])[0],
            'capital-gain': capital_gain,
            'capital-loss': capital_loss,
            'hours-per-week': hours_per_week,
            'native-country': encoders['native-country'].transform([native_country])[0]
        }
        
        input_df = pd.DataFrame([input_data])[X_train.columns]
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        income_label = encoders['annual_income'].inverse_transform([prediction])[0]
        
        high_idx = np.where(encoders['annual_income'].classes_ == '>50K')[0]
        high_income_index = high_idx[0] if len(high_idx) > 0 else 1
        low_income_index = 1 - high_income_index
        
        prob_high = probabilities[high_income_index] * 100
        prob_low = probabilities[low_income_index] * 100
        
        if income_label == '>50K':
            st.markdown(f"""
                <div class="prediction-box-high">
                    <div class="prediction-title">Predicted Income: &gt;50K per year 🎉</div>
                    <div class="prediction-desc">Model Confidence Score: <b>{prob_high:.1f}%</b> probability of earning &gt;50K</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="prediction-box-low">
                    <div class="prediction-title">Predicted Income: &le;50K per year 💵</div>
                    <div class="prediction-desc">Model Confidence Score: <b>{prob_low:.1f}%</b> probability of earning &le;50K</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display Feature Probabilities
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("#### Class Probability Breakdown")
            st.progress(float(probabilities[high_income_index]), text=f">50K Probability: {prob_high:.1f}%")
            st.progress(float(probabilities[low_income_index]), text=f"<=50K Probability: {prob_low:.1f}%")
            
        with res_col2:
            st.markdown("#### Input Summary")
            st.json({
                "Age": age,
                "Education": education,
                "Marital Status": marital_status,
                "Occupation": occupation,
                "Capital Gain": f"${capital_gain:,}",
                "Hours/Week": hours_per_week
            })

# TAB 2: EXPLORATORY DATA ANALYSIS
with tab2:
    st.subheader("Census Dataset Insights & Visualizations")
    
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.markdown("##### Target Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='annual_income', data=df, hue='annual_income', palette=['#38BDF8', '#818CF8'], legend=False, ax=ax)
        ax.set_title("Annual Income Class Counts", fontsize=12, pad=10)
        ax.set_xlabel("Income Class")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        
    with eda_col2:
        st.markdown("##### Age Distribution by Income Class")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.kdeplot(data=df, x='age', hue='annual_income', common_norm=False, palette=['#F43F5E', '#10B981'], fill=True, ax=ax)
        ax.set_title("Age Density Estimation", fontsize=12, pad=10)
        ax.set_xlabel("Age")
        st.pyplot(fig)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    eda_col3, eda_col4 = st.columns(2)
    
    with eda_col3:
        st.markdown("##### Hours Per Week vs Annual Income")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='annual_income', y='hours-per-week', data=df, hue='annual_income', palette=['#6366F1', '#EC4899'], legend=False, ax=ax)
        ax.set_title("Working Hours Distribution", fontsize=12, pad=10)
        ax.set_xlabel("Annual Income")
        ax.set_ylabel("Hours Per Week")
        st.pyplot(fig)
        
    with eda_col4:
        st.markdown("##### Education Num vs High Income (>50K) Proportion")
        edu_income = df.groupby('education-num')['annual_income'].apply(lambda x: (x == '>50K').mean() * 100).reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='education-num', y='annual_income', data=edu_income, hue='education-num', palette='viridis', legend=False, ax=ax)
        ax.set_title("High Income Ratio by Education Years", fontsize=12, pad=10)
        ax.set_xlabel("Education Years (education-num)")
        ax.set_ylabel("% Earning >50K")
        st.pyplot(fig)

# TAB 3: DECISION TREE PERFORMANCE & TREE EXPLORER
with tab3:
    st.subheader("Model Performance & Feature Importance")
    
    eval_col1, eval_col2 = st.columns(2)
    
    with eval_col1:
        st.markdown("##### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=encoders['annual_income'].classes_,
                    yticklabels=encoders['annual_income'].classes_, ax=ax)
        ax.set_title(f"Confusion Matrix (Accuracy: {acc*100:.2f}%)")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)
        
    with eval_col2:
        st.markdown("##### Feature Importances")
        importances = model.feature_importances_
        feature_names = X_train.columns
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(x='Importance', y='Feature', data=feat_df, hue='Feature', palette='Blues_r', legend=False, ax=ax)
        ax.set_title("Decision Tree Feature Importances")
        st.pyplot(fig)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Classification Report Metrics")
    cr_df = pd.DataFrame(cr).transpose()
    st.dataframe(cr_df.style.format("{:.3f}"), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Decision Tree Structure Preview (First 3 Levels)")
    fig, ax = plt.subplots(figsize=(16, 6))
    plot_tree(
        model, 
        max_depth=3, 
        feature_names=X_train.columns, 
        class_names=encoders['annual_income'].classes_,
        filled=True, 
        rounded=True, 
        fontsize=9,
        ax=ax
    )
    st.pyplot(fig)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Census Income Predictor App • Built with Streamlit & Scikit-Learn")
