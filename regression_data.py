import pandas as pd
import numpy as np
import random

np.random.seed(42)

# -----------------------
# Employees
# -----------------------
num_employees = 50
skills = ["Python", "ML", "Web", "Data", "Cloud"]

employees = []

for i in range(num_employees):
    employees.append({
        "employee_id": i,
        "skill": random.choice(skills),
        "experience_level": np.random.randint(1, 6),
        "current_workload": np.random.randint(1, 40),
        "performance_score": round(np.random.uniform(0.5, 1.0), 2)
    })

employees_df = pd.DataFrame(employees)

# -----------------------
# Tasks
# -----------------------
num_tasks = 200

tasks = []

for i in range(num_tasks):
    tasks.append({
        "task_id": i,
        "required_skill": random.choice(skills),
        "difficulty_level": np.random.randint(1, 6),
        "deadline_hours": np.random.randint(1, 72),
        "priority": np.random.randint(1, 4)
    })

tasks_df = pd.DataFrame(tasks)

# -----------------------
# Create Pair Dataset
# -----------------------
rows = []

for _, task in tasks_df.iterrows():
    for _, emp in employees_df.iterrows():

        skill_match = 1 if task["required_skill"] == emp["skill"] else 0

        # Base time depends on difficulty
        base_time = task["difficulty_level"] * 5

        # Efficiency (higher = faster)
        efficiency = emp["experience_level"] * emp["performance_score"]

        # Workload impact
        workload_factor = emp["current_workload"] / 10

        # Final completion time
        completion_time = (
            base_time / (efficiency + 0.1) +
            workload_factor +
            random.uniform(0.5, 2.0)  # noise
        )

        rows.append({
            "task_id": task["task_id"],
            "employee_id": emp["employee_id"],
            "skill_match": skill_match,
            "experience_level": emp["experience_level"],
            "current_workload": emp["current_workload"],
            "performance_score": emp["performance_score"],
            "difficulty_level": task["difficulty_level"],
            "deadline_hours": task["deadline_hours"],
            "priority": task["priority"],
            "completion_time": round(completion_time, 2)
        })

pair_df = pd.DataFrame(rows)

# Save datasets
pair_df.to_csv("regression_dataset.csv", index=False)
employees_df.to_csv("employees.csv", index=False)
tasks_df.to_csv("tasks.csv", index=False)

print("Regression dataset created ✅")
print("Shape:", pair_df.shape)