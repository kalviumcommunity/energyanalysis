import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# STEP 1: Load Data
# -------------------------------
df = pd.read_csv("../processed_data/cleaned_energy_usage.csv")

print("Columns in dataset:", df.columns)

# -------------------------------
# STEP 2: Feature Engineering
# -------------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'])

df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Ensure is_weekend exists
if 'is_weekend' not in df.columns:
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# -------------------------------
# STEP 3: Select Features (SAFE WAY)
# -------------------------------
features = ['hour', 'day_of_week', 'is_weekend']

# Only keep columns that exist
features = [col for col in features if col in df.columns]

X = df[features]
y = df['consumption_kwh']

print("Using features:", features)

# -------------------------------
# STEP 4: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 5: Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# STEP 6: Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# STEP 7: Evaluation
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))

# -------------------------------
# STEP 8: Visualization
# -------------------------------
plt.figure()

plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")

plt.legend()
plt.title("Actual vs Predicted Energy Consumption")

# Save graph
plt.savefig("../output/ml_prediction.png")

plt.show()

print("\nML graph saved in output/ml_prediction.png")