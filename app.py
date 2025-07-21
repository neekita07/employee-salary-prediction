import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_model.pkl")

# Optional background image (using CSS)
st.markdown("""
    <style>
    .main {
        background: url("https://images.unsplash.com/photo-1549924231-f129b911e442?auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .block-container {
       background-color: rgba(30, 30, 30, 0.85); 
       color: #ffffff; 
        padding: 2rem;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# Page settings
st.set_page_config(page_title="Salary Predictor", layout="wide")
st.title("💼 Employee Salary Predictor")
st.markdown("Predict whether an employee earns more than 50K using Machine Learning.")

# Sidebar Input Form
st.sidebar.header("📝 Enter Employee Details")
age = st.sidebar.slider("Age", 18, 75, 30)
workclass = st.sidebar.selectbox("Workclass (Encoded)", list(range(7)))
fnlwgt = st.sidebar.number_input("Fnlwgt (Final Weight)", value=200000)
marital_status = st.sidebar.selectbox("Marital Status (Encoded)", list(range(3)))
occupation = st.sidebar.selectbox("Occupation (Encoded)", list(range(15)))
relationship = st.sidebar.selectbox("Relationship (Encoded)", list(range(6)))
race = st.sidebar.selectbox("Race (Encoded)", list(range(5)))
gender = st.sidebar.selectbox("Gender (0: Female, 1: Male)", [0, 1])
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)
native_country = st.sidebar.selectbox("Native Country (Encoded)", list(range(40)))
educational_num = st.sidebar.slider("Educational Number", 1, 16, 10)

# Prediction Input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country],
    'educational-num': [educational_num]
})

# Centered Prediction Area
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.subheader("🔍 Preview Input")
    st.dataframe(input_df)

    if st.button("🚀 Predict Salary Class"):
        prediction = model.predict(input_df)
        st.success(f"🎯 Prediction: {'>50K' if prediction[0] == '>50K' else '<=50K'}")

    st.markdown("---")
    st.subheader("📂 Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")

    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data:")
        st.dataframe(batch_data.head())
        batch_preds = model.predict(batch_data)
        batch_data['PredictedClass'] = batch_preds
        st.write("✅ Predictions:")
        st.dataframe(batch_data.head())
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", csv, file_name='predicted_output.csv', mime='text/csv')
