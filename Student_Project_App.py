from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "linear_model.pkl"
SCALER_PATH = APP_DIR / "scaler.pkl"

FEATURES = [
    "Attendance",
    "Hours_Studied",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Access_to_Resources_High",
    "Parental_Involvement_High",
    "Gender",
    "School_Type",
    "Sleep_Hours",
]

st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        .app-header {
            border-bottom: 1px solid #e7e9ee;
            padding-bottom: 1rem;
            margin-bottom: 1.4rem;
        }
        .app-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #172033;
            margin: 0;
            line-height: 1.1;
        }
        .app-subtitle {
            color: #596275;
            font-size: 1rem;
            margin-top: 0.45rem;
            max-width: 760px;
        }
        .metric-panel {
            border: 1px solid #e7e9ee;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            background: #ffffff;
        }
        .score-number {
            font-size: 3.2rem;
            font-weight: 850;
            color: #172033;
            line-height: 1;
            margin-bottom: 0.3rem;
        }
        .score-label {
            color: #596275;
            font-size: 0.95rem;
        }
        .status-chip {
            display: inline-block;
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            font-weight: 700;
            font-size: 0.85rem;
            margin-top: 0.75rem;
        }
        .excellent { background: #e8f6ef; color: #1b7f4d; }
        .good { background: #edf4ff; color: #235aa6; }
        .attention { background: #fff4df; color: #956300; }
        .risk { background: #fdecec; color: #b42318; }
        .section-label {
            color: #172033;
            font-size: 1.05rem;
            font-weight: 750;
            margin: 0.2rem 0 0.6rem;
        }
        .small-note {
            color: #697386;
            font-size: 0.9rem;
        }
        div[data-testid="stMetricValue"] {
            color: #172033;
        }
        div.stButton > button:first-child {
            width: 100%;
            border-radius: 8px;
            border: 0;
            background: #172033;
            color: white;
            font-weight: 750;
            padding: 0.7rem 1rem;
        }
        div.stButton > button:first-child:hover {
            background: #26344f;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def yes_no_to_binary(value):
    return 1 if value == "Yes" else 0


def score_band(score):
    if score >= 72:
        return "Excellent", "excellent"
    if score >= 67:
        return "Good", "good"
    if score >= 62:
        return "Needs attention", "attention"
    return "High support needed", "risk"


def build_recommendations(values, score):
    recommendations = []

    if values["Attendance"] < 80:
        recommendations.append("Improve attendance first; it is one of the strongest predictors in this model.")
    if values["Hours_Studied"] < 5:
        recommendations.append("Increase daily focused study time gradually toward 5-7 hours.")
    if values["Previous_Scores"] < 65:
        recommendations.append("Review weak topics from previous exams before adding new material.")
    if values["Tutoring_Sessions"] == 0:
        recommendations.append("Consider at least one tutoring or doubt-clearing session per month.")
    if values["Access_to_Resources_High"] == 0:
        recommendations.append("Improve access to learning resources such as notes, practice papers, and stable internet.")
    if values["Sleep_Hours"] < 6:
        recommendations.append("Bring sleep closer to 7-8 hours to support memory and concentration.")
    if score >= 72:
        recommendations.append("Current inputs suggest strong performance. Maintain consistency and avoid last-minute overloading.")

    return recommendations[:4]


model, scaler = load_artifacts()

st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">Student Exam Score Predictor</h1>
        <div class="app-subtitle">
            Estimate a student's exam score from study habits, academic history, and support factors.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Model Status")
    st.success("Model loaded successfully")
    st.caption("Using the improved 9-feature Ridge model saved from your notebook.")

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="section-label">Student Inputs</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        academic_col, support_col = st.columns(2, gap="large")

        with academic_col:
            st.markdown("#### Academic profile")
            attendance = st.slider("Attendance (%)", 60, 100, 84, 1)
            hours_studied = st.slider("Hours studied per day", 1, 12, 6, 1)
            previous_score = st.slider("Previous exam score", 50, 100, 75, 1)
            sleep_hours = st.slider("Sleep hours per day", 4, 10, 7, 1)
            tutoring_sessions = st.number_input("Tutoring sessions per month", min_value=0, max_value=8, value=2, step=1)

        with support_col:
            st.markdown("#### Environment and support")
            access_resources = st.segmented_control("High access to resources", ["Yes", "No"], default="Yes")
            parent_involvement = st.segmented_control("High parental involvement", ["Yes", "No"], default="Yes")
            gender = st.selectbox("Gender", ["Male", "Female"])
            school_type = st.selectbox("School type", ["Public", "Private"])

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Predict Exam Score")

input_values = {
    "Attendance": attendance,
    "Hours_Studied": hours_studied,
    "Previous_Scores": previous_score,
    "Tutoring_Sessions": tutoring_sessions,
    "Access_to_Resources_High": yes_no_to_binary(access_resources),
    "Parental_Involvement_High": yes_no_to_binary(parent_involvement),
    "Gender": 1 if gender == "Male" else 0,
    "School_Type": 1 if school_type == "Public" else 0,
    "Sleep_Hours": sleep_hours,
}

input_df = pd.DataFrame([input_values], columns=FEATURES)
input_scaled = scaler.transform(input_df)
predicted_score = float(model.predict(input_scaled)[0])
predicted_score = float(np.clip(predicted_score, 0, 100))
band, band_class = score_band(predicted_score)
recommendations = build_recommendations(input_values, predicted_score)

with right:
    st.markdown('<div class="section-label">Prediction Summary</div>', unsafe_allow_html=True)

    if not submitted:
        st.info("Adjust the inputs and click Predict Exam Score to view the result.")

    st.markdown(
        f"""
        <div class="metric-panel">
            <div class="score-label">Predicted exam score</div>
            <div class="score-number">{predicted_score:.2f}</div>
            <div class="score-label">out of 100</div>
            <span class="status-chip {band_class}">{band}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    progress_value = int(round(predicted_score))
    st.progress(progress_value)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Attendance", f"{attendance}%")
    metric_cols[1].metric("Study Hours", f"{hours_studied}/day")
    metric_cols[2].metric("Previous Score", previous_score)

    st.markdown("#### Recommendation Focus")
    for item in recommendations:
        st.write(f"- {item}")

st.divider()

chart_col, data_col = st.columns([1.1, 0.9], gap="large")

with chart_col:
    st.markdown("#### Student Profile Snapshot")
    chart_data = pd.DataFrame(
        {
            "Factor": ["Attendance", "Study Hours", "Previous Score", "Sleep", "Tutoring"],
            "Normalized Value": [
                attendance,
                min(hours_studied / 12 * 100, 100),
                previous_score,
                min(sleep_hours / 10 * 100, 100),
                min(tutoring_sessions / 8 * 100, 100),
            ],
        }
    )
    st.bar_chart(chart_data, x="Factor", y="Normalized Value", height=280)

with data_col:
    st.markdown("#### Model Input Values")
    display_df = input_df.copy()
    display_df["Access_to_Resources_High"] = "Yes" if input_values["Access_to_Resources_High"] else "No"
    display_df["Parental_Involvement_High"] = "Yes" if input_values["Parental_Involvement_High"] else "No"
    display_df["Gender"] = gender
    display_df["School_Type"] = school_type
    display_df = display_df.rename(columns={col: col.replace("_", " ") for col in display_df.columns})
    st.dataframe(display_df.T.rename(columns={0: "Value"}), use_container_width=True)

st.caption("Prediction is based on the trained machine learning model and should be used as decision support, not as a final academic judgment.")

# streamlit run Student_Project_App.py