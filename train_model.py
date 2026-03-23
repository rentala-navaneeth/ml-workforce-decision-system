import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("regression_dataset.csv")

# -----------------------
# Features & Target
# -----------------------
X = df.drop(columns=["completion_time", "task_id", "employee_id"])
y = df["completion_time"]

# -----------------------
# Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Scaling
# -----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# Model
# -----------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# -----------------------
# Predictions
# -----------------------
y_pred = model.predict(X_test_scaled)

# -----------------------
# Evaluation
# -----------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# -----------------------
# Save model & scaler
# -----------------------
joblib.dump(model, "regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved ")