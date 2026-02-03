import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import datetime

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------------------------------------------------------
# BASIC STYLE
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    border-radius: 6px;
    height: 40px;
}
.result-positive {
    background-color: #ffebee;
    padding: 15px;
    border-radius: 8px;
}
.result-negative {
    background-color: #e8f5e9;
    padding: 15px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ['preg','plas','pres','skin','insu','mass','pedi','age','class']
    df = pd.read_csv(url, names=cols)
    df['class'] = df['class'].map({1: 'tested_positive', 0: 'tested_negative'})
    return df

df = load_data()

# -----------------------------------------------------------------------------
# TRAIN MODEL
# -----------------------------------------------------------------------------
@st.cache_resource
def train_model(data):
    X = data[['age','mass','insu','plas']]
    y = data['class']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    return model, acc

model, accuracy = train_model(df)

# -----------------------------------------------------------------------------
# SIDEBAR INPUT
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🩺 Patient Details")

    age = st.number_input("Age", 1, 120, 25)
    bmi = st.number_input("BMI", 0.0, 70.0, 28.0)
    insulin = st.number_input("Insulin", 0, 900, 80)
    glucose = st.number_input("Plasma Glucose", 0, 300, 120)

    predict_btn = st.button("Predict")
    if st.button("Reset"):
        st.rerun()

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
st.title("Diabetes Prediction System")

if predict_btn:
    input_df = pd.DataFrame(
        [[age, bmi, insulin, glucose]],
        columns=['age','mass','insu','plas']
    )

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)

    st.session_state.history.append({
        "Age": age,
        "BMI": bmi,
        "Insulin": insulin,
        "Glucose": glucose,
        "Result": prediction
    })

    if prediction == "tested_positive":
        st.markdown(
            f"<div class='result-positive'><h3>Tested Positive</h3><b>{prob[0][1]*100:.2f}%</b></div>",
            unsafe_allow_html=True
        )
        st.warning("⚠️ Consult a doctor")
    else:
        st.markdown(
            f"<div class='result-negative'><h3>Tested Negative</h3><b>{prob[0][0]*100:.2f}%</b></div>",
            unsafe_allow_html=True
        )
        st.success("✅ Low risk")

    # -----------------------------------------------------------------------------
    # CSV REPORT
    # -----------------------------------------------------------------------------
    report_df = pd.DataFrame({
        "Parameter": ["Age","BMI","Insulin","Glucose","Prediction","Accuracy"],
        "Value": [age,bmi,insulin,glucose,prediction,f"{accuracy:.2%}"]
    })

    st.download_button(
        "Download CSV Report",
        report_df.to_csv(index=False),
        "diabetes_report.csv"
    )

    # -----------------------------------------------------------------------------
    # TXT REPORT
    # -----------------------------------------------------------------------------
    report_txt = f"""
DIABETES PREDICTION REPORT
-------------------------
Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}

Age      : {age}
BMI      : {bmi}
Insulin  : {insulin}
Glucose  : {glucose}

Result   : {prediction}
Accuracy : {accuracy:.2%}

Note: This is a prediction system, not a medical diagnosis.
"""

    st.download_button(
        "Download TXT Report",
        report_txt,
        "diabetes_report.txt"
    )

else:
    st.info("Enter patient data and click Predict")

# -----------------------------------------------------------------------------
# HISTORY
# -----------------------------------------------------------------------------
st.subheader("Prediction History")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.info("No records yet")

st.subheader("Model Accuracy")
st.metric("Accuracy", f"{accuracy:.2%}")
