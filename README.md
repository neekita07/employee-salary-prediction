# 💼 Employee Salary Prediction using Gradient Boosting

Predict whether an employee earns more than 50K using Gradient Boosting Classifier and Streamlit

## 📌 Project Highlights

- 🔍 Predicts employee salary class (`<=50K` or `>50K`) using 13 input features.
- 📊 Data cleaning, encoding, and outlier handling applied.
- 🧠 Gradient Boosting Classifier used as the final model after model comparison.
- 🧪 Real-time single prediction and batch prediction with CSV upload.
- 🚀 Deployed as a modern interactive app using **Streamlit**.

---

## 🧠 ML Pipeline

1. **Data Preprocessing**
   - Handling missing values (`?` replaced with 'Others')
   - Outlier removal in columns like `age`, `capital-gain`, etc.
   - Label Encoding for categorical features

2. **Model Training**
   - Compared multiple models: Logistic Regression, KNN, Random Forest, SVM, Gradient Boosting
   - **Gradient Boosting** showed the highest accuracy and was selected

3. **Deployment**
   - Model saved using `joblib`
   - Streamlit app created with both **single prediction** UI and **batch prediction** CSV upload

---

## 🎯 Features

- 🌟 Input all 13 required features
- 👁️ Preview user inputs before prediction
- 🧾 Batch prediction support with CSV download
- 🎨 Stylish UI with side panel form + central predictions

---

## 💡 Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data manipulation |
| `matplotlib` | Data visualization |
| `sklearn` | ML models and preprocessing |
| `joblib` | Model saving/loading |
| `streamlit` | App interface and deployment |

---

## 🛠 How to Run Locally

1. Clone the repo:
   git clone https://github.com/yourusername/employee-salary-prediction.git
   cd employee-salary-prediction
2. Install dependencies:
   pip install -r requirements.txt
3. Run the Streamlit app:
   streamlit run app.py
