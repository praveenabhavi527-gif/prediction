import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import io

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}
.result-positive {
    background-color: #ffebee;
    padding: 20px;
    border-radius: 10px;
}
.result-negative {
    background-color: #e8f5e9;
    padding: 20px;
    border-radius: 10px;
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
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🩺 Patient Details")

    age = st.number_input("Age", 1, 120, 25)
    mass = st.number_input("BMI", 0.0, 70.0, 28.0)
    insulin = st.number_input("Insulin", 0, 900, 80)
    plasma = st.number_input("Plasma Glucose", 0, 300, 120)

    predict_btn = st.button("🔍 Predict")
    reset_btn = st.button("🔄 Reset")

    if reset_btn:
        st.rerun()

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
st.title("Diabetes Prediction System")

if predict_btn:
    input_df = pd.DataFrame(
        [[age, mass, insulin, plasma]],
        columns=['age','mass','insu','plas']
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)

    st.session_state.history.append({
        "Age": age,
        "BMI": mass,
        "Insulin": insulin,
        "Glucose": plasma,
        "Result": prediction
    })

    # RESULT DISPLAY
    if prediction == "tested_positive":
        st.markdown(
            f"<div class='result-positive'><h2>Tested Positive</h2><h3>{probability[0][1]*100:.2f}%</h3></div>",
            unsafe_allow_html=True
        )
        st.warning("⚠️ Consult a doctor")
    else:
        st.markdown(
            f"<div class='result-negative'><h2>Tested Negative</h2><h3>{probability[0][0]*100:.2f}%</h3></div>",
            unsafe_allow_html=True
        )
        st.success("✅ Low risk")

    # -----------------------------------------------------------------------------
    # CSV REPORT
    # -----------------------------------------------------------------------------
    report_df = pd.DataFrame({
        "Parameter": ["Age", "BMI", "Insulin", "Glucose", "Prediction"],
        "Value": [age, mass, insulin, plasma, prediction]
    })

    st.download_button(
        "📥 Download CSV Report",
        report_df.to_csv(index=False),
        "diabetes_report.csv",
        "text/csv"
    )

    # -----------------------------------------------------------------------------
    # PDF REPORT
    # -----------------------------------------------------------------------------
    def generate_pdf():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Diabetes Prediction Report", styles['Title']))
        elements.append(Spacer(1, 12))

        table_data = [
            ["Parameter", "Value"],
            ["Age", age],
            ["BMI", mass],
            ["Insulin", insulin],
            ["Glucose", plasma],
            ["Prediction", prediction],
            ["Accuracy", f"{accuracy:.2%}"]
        ]

        table = Table(table_data)
        elements.append(table)

        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf_file = generate_pdf()

    st.download_button(
        "📄 Download PDF Report",
        pdf_file,
        "diabetes_report.pdf",
        "application/pdf"
    )

else:
    st.info("👈 Enter values and click Predict")

# -----------------------------------------------------------------------------
# HISTORY
# -----------------------------------------------------------------------------
st.subheader("Prediction History")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.info("No history yet")

# -----------------------------------------------------------------------------
# MODEL INFO
# -----------------------------------------------------------------------------
st.subheader("Model Accuracy")
st.metric("Accuracy", f"{accuracy:.2%}")
