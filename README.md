# Spot-The-Scam
Model Link : eam : https://spot-the-scam-anveshan-hackathon.streamlit.app/

---

# 🕵️‍♀️ Spot the Scam

**Team Name:** Data Tactations
**Problem Statement:** DS-1 — (Spot the Scam)
**Team Members:**

* Amit Singh
* Mamta Yadav

---

## 🔍 Project Overview

Online job platforms are increasingly targeted by scammers. These fake listings waste applicants’ time and put their money and personal information at risk.

Our solution — **Spot the Scam** — is a smart detection system powered by **machine learning** that classifies job listings as **Legit** or **Fraudulent**, and provides real-time visual insights via an interactive **Streamlit dashboard**.

---

## ✨ Key Features & Technologies Used

* ✅ Binary classification model (Logistic Regression / Random Forest)
* ✅ NLP-based feature engineering using TF-IDF
* ✅ SHAP plots for explainability
* ✅ Interactive dashboard (built with Streamlit)
* ✅ Top 5 Legit/Fraud listings with risk alerts
* ✅ CSV Upload for custom prediction

### 🔧 Technologies & Libraries

* Python, Pandas, Scikit-learn
* Plotly, Matplotlib, SHAP
* Streamlit
* Joblib
* Google Colab for model training

---

## 📁 Google Drive Links (Models, Data, etc.)

* 🔗 **Trained Model & Vectorizer**: [View on Google Drive]()
* 🔗 **Dataset (if too large for GitHub)**: [View on Google Drive]()

---

## 📊 Model Performance

* Final Model: **Random Forest Classifier**
* **F1-Score**: *0.92* on test data
* **Evaluation Metrics**: Precision, Recall, Confusion Matrix

---

## 📈 Dashboard Sneak Peek

Includes:

* Overall fraud prediction summary
* Visual charts for fraud vs legit
* Top 5 Fraudulent Jobs (with alerts)
* Top 5 Legitimate Jobs
* Upload section to scan custom job listings

---

## 📹 Video Presentation

🎥 Watch our full walkthrough demo:
🔗 [Click here to view the video]()

---

## 🧠 Data Science Approach

### 1. **Data Cleaning & Processing**

* Removed HTML content, normalized text
* Filled missing values, engineered features

### 2. **Vectorization**

* TF-IDF used for text columns (e.g., title, description)
* Label encoding for categorical fields

### 3. **Modeling**

* Tried Logistic Regression, Random Forest
* Used Stratified Train-Test Split
* Chose best model based on F1-Score

### 4. **Explainability**

* SHAP plots generated for top fraud indicators
* Insights visualized in the dashboard

---

## 🚀 Future Improvements (Nice-to-Haves)

* Real-time API endpoint
* Email alerts for high-risk jobs
* Auto retrain pipeline when new data is added
* Deploy via Streamlit Cloud or Hugging Face Spaces

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE]() file for details.

---

