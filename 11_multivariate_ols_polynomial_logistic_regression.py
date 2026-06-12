import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    classification_report
)

import statsmodels.api as sm

# 1. LOAD DATA
# ==========================================
try:
    energy = pd.read_csv("energy_dataset.csv")
    weather = pd.read_csv("weather_features.csv")
except FileNotFoundError:
    print("Error: CSV files not found. Please check file paths.")
    exit()

# 2. PREPROCESSING
# ==========================================
# Convert timestamps
energy["time"] = pd.to_datetime(energy["time"], utc=True)
weather["dt_iso"] = pd.to_datetime(weather["dt_iso"], utc=True)

# Rename for consistency
energy = energy.rename(columns={"time": "datetime"})
weather = weather.rename(columns={"dt_iso": "datetime"})

# CRITICAL FIX: The weather dataset likely has multiple cities per timestamp.
# We must average them to get one row per hour, otherwise merge will create duplicates.
weather_avg = weather.groupby("datetime")[["temp", "humidity", "clouds_all", "wind_speed"]].mean().reset_index()

# Merge energy and averaged weather data
df = pd.merge(energy, weather_avg, on="datetime", how="inner")

# Define features and target
features = ["temp", "humidity", "clouds_all", "wind_speed"]
target = "total load actual"

# Drop missing values
df = df[[target] + features].dropna()

# 3. MULTIVARIATE LINEAR REGRESSION
# ==========================================
print("\n" + "="*40)
print(" MULTIVARIATE LINEAR REGRESSION")
print("="*40)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print("Coefficients:")
for f, c in zip(features, lr.coef_):
    print(f"  {f}: {c:.4f}")

# Statsmodels for detailed statistics
X_sm = sm.add_constant(X)
ols = sm.OLS(y, X_sm).fit()
print(ols.summary())

# Plots
plt.figure(figsize=(10, 4))

# Scatter Plot: Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual vs Predicted")

# Residual Plot
residuals = y_test - y_pred
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.3, s=10)
plt.axhline(0, color="r", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("linear_regression_plots.png")
plt.show()


# 4. POLYNOMIAL REGRESSION (TEMP ONLY)
# ==========================================
print("\n" + "="*40)
print(" POLYNOMIAL REGRESSION (TEMP ONLY)")
print("="*40)

X_temp = df[["temp"]]  # Keep as DataFrame to maintain 2D shape
y = df[target]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_temp)

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

lr_poly = LinearRegression()
lr_poly.fit(X_train, y_train)

y_pred_poly = lr_poly.predict(X_test)

print(f"R² Score (Temp^2): {r2_score(y_test, y_pred_poly):.4f}")

# Plotting the curve
# Fix: reshape temp_range to (-1, 1) for fit_transform
temp_range = np.linspace(X_temp.min(), X_temp.max(), 300).reshape(-1, 1)
temp_poly_range = poly.transform(temp_range)
y_poly_pred_range = lr_poly.predict(temp_poly_range)

plt.figure(figsize=(8, 6))

plt.scatter(X_temp, y, alpha=0.1, s=2, label="Actual Data")
plt.plot(temp_range, y_poly_pred_range, "r-", lw=2, label="Polynomial Fit (d=2)")
plt.xlabel("Temperature")
plt.ylabel("Energy Demand")
plt.title("Non-Linear Relationship: Temp vs Demand")
plt.legend()
plt.savefig("polynomial_fit.png")
plt.show()


# 5. BINARY LOGISTIC REGRESSION
# ==========================================
print("\n" + "="*40)
print(" BINARY LOGISTIC REGRESSION (High vs Low)")
print("="*40)

# Create binary target (1 if above median, 0 if below)
y_binary = (df[target] > df[target].median()).astype(int)

# Scale features (important for Logistic Regression convergence)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42
)

log_bin = LogisticRegression(max_iter=2000)
log_bin.fit(X_train, y_train)

y_pred = log_bin.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Probability Plot (Sigmoid visualization attempt)
probs = log_bin.predict_proba(X_test)[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(probs, y_test, alpha=0.2, s=10)
plt.xlabel("Predicted Probability of High Demand")
plt.ylabel("Actual Class (0=Low, 1=High)")
plt.title("Logistic Regression Probabilities")
plt.yticks([0, 1])
plt.show()


# 6. MULTICLASS LOGISTIC REGRESSION
# ==========================================
print("\n" + "="*40)
print(" MULTICLASS LOGISTIC REGRESSION (Low, Medium, High)")
print("="*40)

# Create 3 bins (Low, Medium, High) based on quantiles
bins = df[target].quantile([0.33, 0.66]).values
y_multi = pd.cut(
    df[target],
    bins=[-np.inf, bins[0], bins[1], np.inf],
    labels=[0, 1, 2]
).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_multi, test_size=0.2, random_state=42
)

log_multi = LogisticRegression(
    multi_class="multinomial",
    max_iter=3000
)
log_multi.fit(X_train, y_train)

y_pred = log_multi.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))