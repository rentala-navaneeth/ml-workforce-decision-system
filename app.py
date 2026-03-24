import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Load model and data
# -----------------------
import os

if os.path.exists("regression_model.pkl") and os.path.exists("scaler.pkl"):
    model = joblib.load("regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
else:
    from train_model import *
    model = joblib.load("regression_model.pkl")
    scaler = joblib.load("scaler.pkl")

employees_df = pd.read_csv("employees.csv")

# -----------------------
# UI Title
# -----------------------
st.title("ML-Powered Workforce Decision Automation System")

st.write("Predict task completion time and assign optimal employee")

# -----------------------
# Inputs
# -----------------------
skill = st.selectbox("Required Skill", ["Python", "ML", "Web", "Data", "Cloud"])
difficulty = st.slider("Task Difficulty", 1, 5, 3)
deadline = st.number_input("Deadline (hours)", 1, 100, 24)
priority = st.slider("Priority", 1, 3, 2)

# -----------------------
# Run Model
# -----------------------
if st.button("Optimize Assignment"):

    results = []

    for _, emp in employees_df.iterrows():

        skill_match = 1 if skill == emp["skill"] else 0

        features = pd.DataFrame([{
            "skill_match": skill_match,
            "experience_level": emp["experience_level"],
            "current_workload": emp["current_workload"],
            "performance_score": emp["performance_score"],
            "difficulty_level": difficulty,
            "deadline_hours": deadline,
            "priority": priority
        }])

        features_scaled = scaler.transform(features)
        predicted_time = model.predict(features_scaled)[0]

        results.append({
            "employee_id": int(emp["employee_id"]),
            "predicted_time": round(predicted_time, 2)
        })

    results_df = pd.DataFrame(results)

    # -----------------------
    # Best employee
    # -----------------------
    best = results_df.sort_values(by="predicted_time").iloc[0]

    st.subheader("Best Assignment")
    st.write(f"Employee {int(best['employee_id'])} → Predicted Time: {best['predicted_time']} hrs")
    # -----------------------
    # Top 3
    # -----------------------
    st.subheader("Top 3 Employees")

    top3 = results_df.sort_values(by="predicted_time").head(3)

    for i, row in top3.reset_index(drop=True).iterrows():
        st.write(f"Rank {i+1}: Employee {int(row['employee_id'])} (Time: {row['predicted_time']} hrs)")
    # -----------------------
    # Baseline comparison
    # -----------------------
    import random

    random_emp = employees_df.sample(1).iloc[0]

    skill_match = 1 if skill == random_emp["skill"] else 0

    baseline_features = pd.DataFrame([{
        "skill_match": skill_match,
        "experience_level": random_emp["experience_level"],
        "current_workload": random_emp["current_workload"],
        "performance_score": random_emp["performance_score"],
        "difficulty_level": difficulty,
        "deadline_hours": deadline,
        "priority": priority
    }])

    baseline_scaled = scaler.transform(baseline_features)
    baseline_time = model.predict(baseline_scaled)[0]

    improvement = ((baseline_time - best["predicted_time"]) / baseline_time) * 100

    st.subheader("Optimization Impact")

    st.write(f"Random Assignment Time: {round(baseline_time,2)} hrs")
    st.write(f"Optimized Time: {best['predicted_time']} hrs")
    st.write(f"Improvement: {round(improvement,2)}%")