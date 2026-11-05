import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load trained models
grade_model = joblib.load("models/grade_model.joblib")
risk_model = joblib.load("models/risk_model.joblib")

st.title("🎓 Student Performance Prediction System")
st.write(
    "This application predicts a student's expected final grade and academic "
    "risk level using study habits, attendance, family background, and school data."
)

st.sidebar.header("Enter Student Details")

# Personal information
school = st.sidebar.selectbox("School", ["GP", "MS"])
age = st.sidebar.slider("Age", 15, 22, 17)
sex = st.sidebar.selectbox("Sex", ["F", "M"])
address = st.sidebar.selectbox("Home Address", ["U", "R"])
famsize = st.sidebar.selectbox("Family Size", ["LE3", "GT3"])
Pstatus = st.sidebar.selectbox("Parents' Status", ["T", "A"])

# Parent information
Medu = st.sidebar.slider("Mother's Education (0-4)", 0, 4, 2)
Fedu = st.sidebar.slider("Father's Education (0-4)", 0, 4, 2)

Mjob = st.sidebar.selectbox(
    "Mother's Job",
    ["teacher", "health", "services", "at_home", "other"]
)

Fjob = st.sidebar.selectbox(
    "Father's Job",
    ["teacher", "health", "services", "at_home", "other"]
)

guardian = st.sidebar.selectbox("Guardian", ["mother", "father", "other"])

# School choice and travel
reason = st.sidebar.selectbox(
    "Reason to Choose School",
    ["home", "reputation", "course", "other"]
)
traveltime = st.sidebar.slider("Travel Time to School (1-4)", 1, 4, 1)

# Academic information
studytime = st.sidebar.slider("Weekly Study Time (1-4)", 1, 4, 2)
failures = st.sidebar.slider("Past Class Failures", 0, 3, 0)
absences = st.sidebar.slider("School Absences", 0, 93, 4)

schoolsup = st.sidebar.selectbox("Extra Educational Support", ["yes", "no"])
famsup = st.sidebar.selectbox("Family Educational Support", ["yes", "no"])
paid = st.sidebar.selectbox("Extra Paid Classes", ["yes", "no"])
activities = st.sidebar.selectbox("Extracurricular Activities", ["yes", "no"])
nursery = st.sidebar.selectbox("Attended Nursery School", ["yes", "no"])
higher = st.sidebar.selectbox("Wants Higher Education", ["yes", "no"])
internet = st.sidebar.selectbox("Internet Access", ["yes", "no"])
romantic = st.sidebar.selectbox("In Romantic Relationship", ["yes", "no"])

# Lifestyle and health
famrel = st.sidebar.slider("Family Relationship Quality (1-5)", 1, 5, 4)
freetime = st.sidebar.slider("Free Time After School (1-5)", 1, 5, 3)
goout = st.sidebar.slider("Going Out With Friends (1-5)", 1, 5, 3)
Dalc = st.sidebar.slider("Workday Alcohol Consumption (1-5)", 1, 5, 1)
Walc = st.sidebar.slider("Weekend Alcohol Consumption (1-5)", 1, 5, 1)
health = st.sidebar.slider("Current Health Status (1-5)", 1, 5, 3)

# Create input data
input_data = pd.DataFrame([{
    "school": school,
    "sex": sex,
    "age": age,
    "address": address,
    "famsize": famsize,
    "Pstatus": Pstatus,
    "Medu": Medu,
    "Fedu": Fedu,
    "Mjob": Mjob,
    "Fjob": Fjob,
    "reason": reason,
    "guardian": guardian,
    "traveltime": traveltime,
    "studytime": studytime,
    "failures": failures,
    "absences": absences,
    "schoolsup": schoolsup,
    "famsup": famsup,
    "paid": paid,
    "activities": activities,
    "nursery": nursery,
    "higher": higher,
    "internet": internet,
    "romantic": romantic,
    "famrel": famrel,
    "freetime": freetime,
    "goout": goout,
    "Dalc": Dalc,
    "Walc": Walc,
    "health": health
}])

# Add engineered features used during training
input_data["attendance_score"] = 1 - (input_data["absences"] / 93)
input_data["failure_impact"] = input_data["failures"] * input_data["absences"]
input_data["study_support_score"] = (
    input_data["studytime"]
    + input_data["famsup"].map({"yes": 1, "no": 0})
)

# Safety check: make sure both models get exactly the columns they were trained on
for name, mdl in [("grade_model", grade_model), ("risk_model", risk_model)]:
    expected = getattr(mdl, "feature_names_in_", None)
    if expected is not None:
        missing = [c for c in expected if c not in input_data.columns]
        if missing:
            st.error(f"{name} expects columns the app does not provide: {missing}")
            st.stop()

st.subheader("📋 Student Details")
st.dataframe(input_data.T.rename(columns={0: "Value"}))

if st.button("Predict Performance"):

    predicted_grade = grade_model.predict(input_data)[0]
    risk_prediction = risk_model.predict(input_data)[0]
    risk_probability = risk_model.predict_proba(input_data)[0]

    probability_df = pd.DataFrame({
        "Risk Level": risk_model.classes_,
        "Probability": risk_probability
    })

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Final Grade",
            value=f"{predicted_grade:.1f} / 20"
        )

    with col2:
        if risk_prediction == "Low":
            st.success("Low Academic Risk")
        elif risk_prediction == "Medium":
            st.warning("Medium Academic Risk")
        else:
            st.error("High Academic Risk")

    st.subheader("Risk Probability")
    st.bar_chart(probability_df.set_index("Risk Level"))

    st.subheader("Recommendation")

    if risk_prediction == "High":
        st.write(
            "- Arrange academic counselling.\n"
            "- Monitor attendance and assignment completion.\n"
            "- Provide additional learning support."
        )
    elif risk_prediction == "Medium":
        st.write(
            "- Track study time and attendance regularly.\n"
            "- Encourage participation in revision sessions.\n"
            "- Review performance after the next assessment."
        )
    else:
        st.write(
            "- Maintain the current study routine.\n"
            "- Continue regular academic monitoring."
        )