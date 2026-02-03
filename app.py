import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 50px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #45a049;
}
.result-card {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-top: 20px;
}
.result-positive {
    background-color: #ffebee;
    color: #c62828;
    border: 1px solid #ef9a9a;
}
.result-negative {
    background-color: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #a5d6a7;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE (History)
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age', 'class']
    df = pd.read_csv(url, names=columns)
    df['class'] = df['class'].map({1: 'tested_positive', 0: 'tested_negative'})
    return df

df = load_data()

# -----------------------------------------------------------------------------
# TRAIN MODEL
# -----------------------------------------------------------------------------
@st.cache_resource
def train_model(data):
    X = data[['age', 'mass', 'insu', 'plas']]
    y = data['class']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    return model, acc

model, accuracy = train_model(df)

# -----------------------------------------------------------------------------
# SIDEBAR INPUTS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🩺 Patient Details")

    age = st.number_input("Age", 1, 120, 25)
    mass = st.number_input("BMI", 0.0, 70.0, 28.0)
    insulin = st.number_input("Insulin Level", 0, 900, 80)
    plasma = st.number_input("Plasma Glucose", 0, 300, 120)

    predict_btn = st.button("🔍 Analyze Result")
    reset_btn = st.button("🔄 Reset")

    if reset_btn:
        st.experimental_rerun()

# -----------------------------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------------------------
st.title("Diabetes Prediction System")
st.write("Predicts diabetes using **Logistic Regression**.")

col1, col2 = st.columns([2, 1])

with col1:
    if predict_btn:
        input_df = pd.DataFrame(
            [[age, mass, insulin, plasma]],
            columns=['age', 'mass', 'insu', 'plas']
        )

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)

        # Save history
        st.session_state.history.append({
            "Age": age,
            "BMI": mass,
            "Insulin": insulin,
            "Glucose": plasma,
            "Result": prediction
        })

        # Display result
        if prediction == "tested_positive":
            score = probability[0][1] * 100
            st.markdown(f"""
            <div class="result-card result-positive">
                <h2>Tested Positive</h2>
                <h1>{score:.2f}%</h1>
                <p>High risk of diabetes</p>
            </div>
            """, unsafe_allow_html=True)
            st.warning("⚠️ Please consult a doctor.")
        else:
            score = probability[0][0] * 100
            st.markdown(f"""
            <div class="result-card result-negative">
                <h2>Tested Negative</h2>
                <h1>{score:.2f}%</h1>
                <p>Low risk of diabetes</p>
            </div>
            """, unsafe_allow_html=True)
            st.success("✅ Maintain a healthy lifestyle.")

        # Probability chart
        prob_df = pd.DataFrame({
            "Outcome": ["Negative", "Positive"],
            "Probability (%)": [probability[0][0]*100, probability[0][1]*100]
        })
        st.subheader("Prediction Probability")
        st.bar_chart(prob_df.set_index("Outcome"))

        # Download report
        report = pd.DataFrame({
            "Parameter": ["Age", "BMI", "Insulin", "Glucose", "Result"],
            "Value": [age, mass, insulin, plasma, prediction]
        })

        st.download_button(
            "📥 Download Report",
            report.to_csv(index=False),
            "diabetes_report.csv",
            "text/csv"
        )
    else:
        st.info("👈 Enter values and click Analyze Result")

with col2:
    st.subheader("Model Performance")
    st.metric("Accuracy", f"{accuracy:.2%}")
    st.progress(accuracy)

    st.subheader("Features Used")
    st.code("['age', 'mass', 'insu', 'plas']")

# -----------------------------------------------------------------------------
# HISTORY TABLE
# -----------------------------------------------------------------------------
st.subheader("Patient Prediction History")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.info("No records yet.")
