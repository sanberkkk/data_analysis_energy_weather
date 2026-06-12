import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.signal import correlate
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
df = pd.read_csv("spain_energy_weather_hourly.csv")
df = df[['time', 'temp_avg', 'total load actual']].dropna()

# Convert time column and sort
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')

# Ensure alignment and normalize for cross-correlation
temp = df['temp_avg'] - df['temp_avg'].mean()
demand = df['total load actual'] - df['total load actual'].mean()

# --- Cross-correlation plot ---
lags = range(-24, 25)
ccf = [correlate(temp, demand.shift(lag).fillna(0)).mean() for lag in lags]

plt.figure(figsize=(10, 5))
sns.lineplot(x=lags, y=ccf, marker="o")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.title("Cross-Correlation: Temperature vs. Energy Demand")
plt.xlabel("Lag (hours) — Positive = Temp leads")
plt.ylabel("Cross-Correlation")
plt.grid(True)
plt.tight_layout()
plt.savefig("cross_correlation_temp_demand.png")
plt.show()

# --- Lagged Linear Regression (e.g., temp leads by 3h) ---
lag_hours = 3
df['temp_lagged'] = df['temp_avg'].shift(lag_hours)
reg_df = df.dropna()

X = reg_df[['temp_lagged']]
y = reg_df['total load actual']
model = LinearRegression().fit(X, y)

r2 = model.score(X, y)
print(f"\nLagged Regression (Temp leads by {lag_hours}h):")
print(f"Regression Equation: Demand = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Temp (lagged)")
print(f"R² = {r2:.4f}")

# --- Granger Causality Test ---
# Must be two-column with no NaNs
gc_df = df[['total load actual', 'temp_avg']].dropna()
print("\nGranger Causality Test (Does Temp 'cause' Demand):")
grangercausalitytests(gc_df, maxlag=6, verbose=True)
