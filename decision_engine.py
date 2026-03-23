import pandas as pd
import joblib

# -----------------------
# Load model and scaler
# -----------------------
model = joblib.load("regression_model.pkl")
scaler = joblib.load("scaler.pkl")

employees_df = pd.read_csv("employees.csv")
tasks_df = pd.read_csv("tasks.csv")

# -----------------------
# Decision logic
# -----------------------
results = []

for _, task in tasks_df.iterrows():

    best_employee = None
    best_time = float("inf")

    for _, emp in employees_df.iterrows():

        skill_match = 1 if task["required_skill"] == emp["skill"] else 0

        features = pd.DataFrame([{
            "skill_match": skill_match,
            "experience_level": emp["experience_level"],
            "current_workload": emp["current_workload"],
            "performance_score": emp["performance_score"],
            "difficulty_level": task["difficulty_level"],
            "deadline_hours": task["deadline_hours"],
            "priority": task["priority"]
        }])

        features_scaled = scaler.transform(features)
        predicted_time = model.predict(features_scaled)[0]

        if predicted_time < best_time:
            best_time = predicted_time
            best_employee = int(emp["employee_id"])

    results.append({
        "task_id": int(task["task_id"]),
        "best_employee": best_employee,
        "predicted_time": round(best_time, 2)
    })

results_df = pd.DataFrame(results)

print("\nOptimized Task Assignments:")
print(results_df.head())

import random

# -----------------------
# BASELINE (Random Assignment)
# -----------------------
baseline_times = []

for _, task in tasks_df.iterrows():

    random_emp = employees_df.sample(1).iloc[0]

    skill_match = 1 if task["required_skill"] == random_emp["skill"] else 0

    features = pd.DataFrame([{
        "skill_match": skill_match,
        "experience_level": random_emp["experience_level"],
        "current_workload": random_emp["current_workload"],
        "performance_score": random_emp["performance_score"],
        "difficulty_level": task["difficulty_level"],
        "deadline_hours": task["deadline_hours"],
        "priority": task["priority"]
    }])

    features_scaled = scaler.transform(features)
    predicted_time = model.predict(features_scaled)[0]

    baseline_times.append(predicted_time)

# -----------------------
# OPTIMIZED (Your System)
# -----------------------
optimized_times = results_df["predicted_time"]

# -----------------------
# COMPARISON
# -----------------------
baseline_avg = sum(baseline_times) / len(baseline_times)
optimized_avg = sum(optimized_times) / len(optimized_times)

improvement = ((baseline_avg - optimized_avg) / baseline_avg) * 100

print("\n--- Performance Comparison ---")
print(f"Baseline Avg Time: {baseline_avg:.2f}")
print(f"Optimized Avg Time: {optimized_avg:.2f}")
print(f"Improvement: {improvement:.2f}%")