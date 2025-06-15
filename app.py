import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Set up the page
st.set_page_config(page_title="Spot the Scam", layout="wide")

# Custom CSS for design and animation
st.markdown("""
    <style>
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        .main-title {
            color: #d72638;
            font-size: 48px;
            text-align: center;
            font-weight: bold;
            animation: fadeIn 1s ease-in-out;
        }
        .sub-section {
            background-color: #fef2f2;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            animation: fadeIn 0.8s ease-in-out;
            color: #2c2c2c;
        }
        .apply-button {
            background-color: #008000;
            color: white;
            padding: 6px 12px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .apply-button:hover {
            background-color: #005c00;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main-title'>🚨 Spot the Scam: Job Fraud Detection</div>", unsafe_allow_html=True)

# Sidebar Description + Training Scores
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/6197/6197833.png", width=100)
    st.markdown("### 🤖 About the Model", unsafe_allow_html=True)
    st.markdown("""
    <div style='color: #f5f5f5; font-size: 15px; line-height: 1.6;'>
    This app uses <strong>machine learning</strong> to detect fake job listings that could waste users' time or steal personal information.<br><br>
    It predicts fraud risk based on job title and description, and shows:
    <ul>
        <li>🔍 <strong>Top high-risk listings</strong></li>
        <li>✅ <strong>Legit jobs</strong></li>
        <li>📉 <strong>Fraud probabilities</strong></li>
        <li>📊 <strong>Pie & histogram charts</strong></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Training Metrics")
    st.success("**Accuracy:** 0.9342")
    st.success("**Precision:** 0.9203")
    st.success("**Recall:** 0.9455")
    st.success("**F1 Score:** 0.9327")

    st.markdown("---")
    st.markdown("📤 Upload a CSV below to begin scanning job listings:")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload job listings (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")

    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    X = vectorizer.transform(df['text'])

    df['fraud_prob'] = model.predict_proba(X)[:, 1]
    df['prediction'] = model.predict(X)
    df['label'] = df['prediction'].map({0: "✅ Genuine", 1: "❌ Fraudulent"})

    total = len(df)
    frauds = df['prediction'].sum()

    st.markdown(f"""
        <div class='sub-section'>
            <h3>📋 Summary</h3>
            <p><strong>Total Jobs:</strong> {total} &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
            <strong>Genuine:</strong> {total - frauds} &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
            <strong>Fraudulent:</strong> {frauds}</p>
        </div>
    """, unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        pie = px.pie(df, names='label', title='Fraudulent vs Genuine Jobs', color_discrete_sequence=['#ff6361', '#58508d'])
        st.plotly_chart(pie, use_container_width=True)

    with col2:
        hist = px.histogram(df, x='fraud_prob', nbins=20, title='Fraud Probability Distribution', color_discrete_sequence=['#ffa600'])
        st.plotly_chart(hist, use_container_width=True)

    # Apply Button Logic
    def apply_job(row):
        if row['prediction'] == 1:
            st.warning("🚫 This job is flagged as FRAUDULENT. Application blocked.")
        else:
            with st.form(f"apply_form_{row.name}"):
                st.success("✅ This job looks genuine! Enter details to proceed.")
                email = st.text_input("Email", key=f"email_{row.name}")
                phone = st.text_input("Phone", key=f"phone_{row.name}")
                if st.form_submit_button("Submit Application"):
                    st.success(f"✅ Application submitted for: {row['title']}")

    # Show full table
    st.markdown("<h3>📄 All Job Listings</h3>", unsafe_allow_html=True)
    st.dataframe(df[['title', 'location', 'fraud_prob', 'label']])

    # Top 5 Fraudulent Jobs
    st.markdown("<h3>❌ Top 5 Most Fraudulent Jobs</h3>", unsafe_allow_html=True)
    top_fraud = df[df['prediction'] == 1].sort_values(by='fraud_prob', ascending=False).head(5)
    for i, row in top_fraud.iterrows():
        st.markdown(f"""
            <div class='sub-section'>
                <strong>Title:</strong> {row['title']}<br>
                <strong>Location:</strong> {row['location']}<br>
                <strong>Fraud Probability:</strong> {row['fraud_prob']:.2%}<br>
                <strong>Status:</strong> {row['label']}<br>
            </div>
        """, unsafe_allow_html=True)
        apply_job(row)

    # Top 5 Legitimate Jobs
    st.markdown("<h3>✅ Top 5 Legitimate Jobs</h3>", unsafe_allow_html=True)
    legit_jobs = df[df['prediction'] == 0].sort_values(by='fraud_prob', ascending=False).head(5)
    for i, row in legit_jobs.iterrows():
        st.markdown(f"""
            <div class='sub-section'>
                <strong>Title:</strong> {row['title']}<br>
                <strong>Location:</strong> {row['location']}<br>
                <strong>Fraud Probability:</strong> {row['fraud_prob']:.2%}<br>
                <strong>Status:</strong> {row['label']}<br>
            </div>
        """, unsafe_allow_html=True)
        apply_job(row)
