import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

# --- Mapping dictionaries for user-friendly dropdowns ---
workclass_options = {
    "Private": 0,
    "Self-emp-not-inc": 1,
    "Self-emp-inc": 2,
    "Federal-gov": 3,
    "Local-gov": 4,
    "State-gov": 5,
    "Without-pay": 6,
    "Never-worked": 7
}
marital_status_options = {
    "Married-civ-spouse": 0,
    "Divorced": 1,
    "Never-married": 2,
    "Separated": 3,
    "Widowed": 4,
    "Married-spouse-absent": 5
}
occupation_options = {
    "Tech-support": 0,
    "Craft-repair": 1,
    "Other-service": 2,
    "Sales": 3,
    "Exec-managerial": 4,
    "Prof-specialty": 5,
    "Handlers-cleaners": 6,
    "Machine-op-inspct": 7,
    "Adm-clerical": 8,
    "Farming-fishing": 9,
    "Transport-moving": 10,
    "Priv-house-serv": 11,
    "Protective-serv": 12,
    "Armed-Forces": 13
}
relationship_options = {
    "Wife": 0,
    "Own-child": 1,
    "Husband": 2,
    "Not-in-family": 3,
    "Other-relative": 4,
    "Unmarried": 5
}
race_options = {
    "White": 0,
    "Asian-Pac-Islander": 1,
    "Amer-Indian-Eskimo": 2,
    "Other": 3,
    "Black": 4
}
gender_options = {"Male": 0, "Female": 1}
native_country_options = {
    "United-States": 0,
    "Cambodia": 1,
    "England": 2,
    "Puerto-Rico": 3,
    "Canada": 4,
    "Germany": 5,
    "Outlying-US(Guam-USVI-etc)": 6,
    "India": 7,
    "Japan": 8,
    "Greece": 9,
    "South": 10,
    "China": 11,
    "Cuba": 12,
    "Iran": 13,
    "Honduras": 14,
    "Philippines": 15,
    "Italy": 16,
    "Poland": 17,
    "Jamaica": 18,
    "Vietnam": 19,
    "Mexico": 20,
    "Portugal": 21,
    "Ireland": 22,
    "France": 23,
    "Dominican-Republic": 24,
    "Laos": 25,
    "Ecuador": 26,
    "Taiwan": 27,
    "Haiti": 28,
    "Columbia": 29,
    "Hungary": 30,
    "Guatemala": 31,
    "Nicaragua": 32,
    "Scotland": 33,
    "Thailand": 34,
    "Yugoslavia": 35,
    "El-Salvador": 36,
    "Trinadad&Tobago": 37,
    "Peru": 38,
    "Hong": 39
}

# Optional background styling
st.markdown("""
    <style>
    .block-container {
       background-color: rgba(30, 30, 30, 0.85); 
       color: #ffffff; 
        padding: 2rem;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Layout ---
st.title("💼 Employee Salary Predictor")
st.markdown("Predict whether an employee earns more than 50K using Machine Learning.")

col1, col2 = st.columns([1, 2])

with col1:
    st.sidebar.header("📝 Enter Employee Details")
    age = st.sidebar.slider("Age", 18, 75, 30)
    workclass = workclass_options[st.sidebar.selectbox("Workclass", list(workclass_options.keys()))]
    fnlwgt = st.sidebar.number_input("Fnlwgt (Final Weight)", value=200000)
    marital_status = marital_status_options[st.sidebar.selectbox("Marital Status", list(marital_status_options.keys()))]
    occupation = occupation_options[st.sidebar.selectbox("Occupation", list(occupation_options.keys()))]
    relationship = relationship_options[st.sidebar.selectbox("Relationship", list(relationship_options.keys()))]
    race = race_options[st.sidebar.selectbox("Race", list(race_options.keys()))]
    gender = gender_options[st.sidebar.selectbox("Gender", list(gender_options.keys()))]
    capital_gain = st.sidebar.number_input("Capital Gain", value=0)
    capital_loss = st.sidebar.number_input("Capital Loss", value=0)
    hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)
    native_country = native_country_options[st.sidebar.selectbox("Native Country", list(native_country_options.keys()))]
    educational_num = st.sidebar.slider("Educational Number", 1, 16, 10)

with col2:
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

    st.subheader("🔍 Preview Input")
    st.write(input_df)

    if st.button("🚀 Predict Salary Class"):
        prediction = model.predict(input_df)
        st.success(f"✅ Prediction: {'>50K' if prediction[0] == '>50K' else '<=50K'}")

    # Batch prediction
    st.markdown("---")
    st.subheader("📂 Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", batch_df.head())
        batch_preds = model.predict(batch_df)
        batch_df['Predicted'] = batch_preds
        st.write("✅ Predictions:", batch_df.head())
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, file_name="batch_predictions.csv", mime="text/csv")
