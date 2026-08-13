import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor", page_icon="🎓", layout="wide"
)


# Load the trained pickle model using exact directory path
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model.pkl")

    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()

# Mappings for Categorical Features (Convert text to numbers)
gender_map = {"Male": 0, "Female": 1, "Other": 2}

department_map = {
    "Arts": 0,
    "Business": 1,
    "Computer Science": 2,
    "Engineering": 3,
    "Science": 4,
}

# Header Section
st.title("🎓 Student Performance Prediction App")
st.markdown(
    "Enter the student's demographic and academic details below to predict whether they will **Pass** or **Fail**."
)
st.divider()

# Form layout split into two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Demographic & General Info")
    age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
    gender_input = st.selectbox("Gender", options=list(gender_map.keys()))
    department_input = st.selectbox(
        "Department", options=list(department_map.keys())
    )

with col2:
    st.subheader("📚 Academic Metrics")
    study_hours = st.number_input(
        "Study Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=4.0,
        step=0.5,
    )
    attendance = st.slider(
        "Attendance Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0,
    )
    assignments = st.number_input(
        "Assignments Completed", min_value=0, max_value=100, value=10, step=1
    )
    midterm_score = st.number_input(
        "Midterm Score", min_value=0.0, max_value=100.0, value=75.0, step=1.0
    )
    final_score = st.number_input(
        "Final Score", min_value=0.0, max_value=100.0, value=78.0, step=1.0
    )

st.divider()

# Prediction Button
if st.button("🔮 Predict Result", type="primary", use_container_width=True):
    # Convert string choices to numeric values expected by the model
    gender_encoded = gender_map[gender_input]
    department_encoded = department_map[department_input]

    # Construct input dataframe with encoded values
    input_data = pd.DataFrame([
        {
            "Age": age,
            "Gender": gender_encoded,
            "Department": department_encoded,
            "Study_Hours_Per_Day": study_hours,
            "Attendance_Percentage": attendance,
            "Assignments_Completed": assignments,
            "Midterm_Score": midterm_score,
            "Final_Score": final_score,
        }
    ])

    try:
        # Make Prediction
        prediction = model.predict(input_data)[0]

        # Display Result
        st.subheader("Prediction Result:")
        if str(prediction).strip().lower() == "pass":
            st.success("🎉 **Result:** The student is predicted to **PASS**!")
        else:
            st.error("⚠️ **Result:** The student is predicted to **FAIL**.")

        # Show prediction probability if supported
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            st.write("### Prediction Confidence")
            prob_df = pd.DataFrame({
                "Outcome": classes,
                "Probability": [f"{p * 100:.2f}%" for p in probabilities],
            })
            st.dataframe(prob_df, hide_index=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
