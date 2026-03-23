# ML-Powered Workforce Decision Automation System

## Overview
This project is an ML-powered decision automation system that optimizes workforce task allocation by predicting task completion time and selecting the most efficient employee. 

Unlike traditional ML projects that focus only on prediction, this system integrates regression modeling with decision optimization to reduce overall task completion time.
---

## Problem Statement
Traditional task assignment methods often lead to inefficient workload distribution and increased completion times due to lack of data-driven decision making.

---

## Solution
This system uses a regression-based machine learning model to:
- Predict task completion time for each employee
- Evaluate all possible assignments
- Select the employee with the minimum predicted completion time
- Quantify improvement over baseline (random assignment)

---

## System Architecture

Data Layer → Feature Engineering → Regression Model → Prediction Layer  
→ Decision Engine (Optimization) → Performance Comparison (Baseline vs ML)

## Dataset
Synthetic dataset simulating a workforce environment:

### Employee Features:
- skill
- experience_level
- current_workload
- performance_score

### Task Features:
- required_skill
- difficulty_level
- deadline_hours
- priority

### Target:
- completion_time (continuous)

---

## Model
- Algorithm: Random Forest Regressor
- Preprocessing: StandardScaler
- Train-Test Split: 80/20


## Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score


## Results
- RMSE: ~0.49
- MAE: ~0.41
- R² Score: ~0.99

## Why It Works
The model captures relationships between task difficulty, employee efficiency, and workload. 

By evaluating all employee-task combinations, the system identifies the assignment that minimizes completion time, leading to significant efficiency gains compared to random allocation.

## Decision Optimization
For each task:
- Predict completion time for all employees
- Select employee with minimum predicted time

---

## Performance Improvement
- Baseline (Random Assignment): 11.83 hrs
- Optimized Assignment: 5.60 hrs
- Improvement: **52.7% reduction in completion time**

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## How to Run
### 1. Install dependencies

pip install -r requirements.txt


### 2. Generate dataset

python regression_data.py


### 3. Train model

python train_model.py


### 4. Run decision engine

python decision_engine.py


### 5. Launch UI

streamlit run app.py

---

## Key Highlights
- Built an end-to-end ML pipeline (data → model → decision)
- Implemented regression-based prediction for task completion time
- Designed a decision automation layer for optimal task assignment
- Achieved measurable improvement using ML-based optimization
- Developed an interactive UI for real-time decision making

---

## Future Improvements
- Use real-world datasets
- Add advanced models (XGBoost, LightGBM)
- Deploy as API-based system
- Add feedback loop for continuous learning

## Use Cases
- IT ticket assignment systems
- Workforce scheduling in organizations
- Customer support routing
- Resource allocation in operations