import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load datasets
energy = pd.read_csv("energy_dataset.csv")
weather = pd.read_csv("weather_features.csv")

# Datetime processing
energy['time'] = pd.to_datetime(energy['time'], errors='coerce', utc=True)
weather['dt_iso'] = pd.to_datetime(weather['dt_iso'].str.slice(0, 19), errors='coerce', utc=True)

# Average wind speed (hourly)
weather_avg = weather.groupby('dt_iso')['wind_speed'].mean().reset_index()
energy['time_rounded'] = energy['time'].dt.round('h')
weather_avg['dt_rounded'] = weather_avg['dt_iso'].dt.round('h')

# Merge
merged = pd.merge(energy, weather_avg, left_on='time_rounded', right_on='dt_rounded', how='inner')
data = merged.dropna(subset=['wind_speed', 'generation wind onshore', 'generation wind offshore'])

# Extract variables
speed = data['wind_speed']
onshore = data['generation wind onshore']
offshore = data['generation wind offshore']

# Correlation
p1, pp1 = pearsonr(speed, onshore)
s1, sp1 = spearmanr(speed, onshore)
try:
    p2, pp2 = pearsonr(speed, offshore)
    s2, sp2 = spearmanr(speed, offshore)
except:
    p2 = pp2 = s2 = sp2 = float('nan')

print("=== Wind Speed vs Onshore Wind Generation ===")
print(f"Pearson r = {p1:.4f} (p = {pp1:.4f})")
print(f"Spearman r = {s1:.4f} (p = {sp1:.4f})")
if pp1 < 0.05:
    print("➤ Reject H₀: Wind speed is correlated with onshore wind generation.")
else:
    print("➤ Fail to reject H₀.")

print("\n=== Wind Speed vs Offshore Wind Generation ===")
print(f"Pearson r = {p2}, Spearman r = {s2}")
print("➤ Cannot test H₀ due to lack of variation in offshore generation.\n")

# Covariance
print("Covariance matrix:")
print(data[['wind_speed', 'generation wind onshore', 'generation wind offshore']].cov())

# Linear regression
X = speed.values.reshape(-1, 1)
y = onshore.values
reg = LinearRegression().fit(X, y)
slope = reg.coef_[0]
intercept = reg.intercept_
r2 = reg.score(X, y)

print(f"\n=== Linear Regression ===")
print(f"Regression equation: y = {intercept:.2f} + {slope:.2f} * wind_speed")
print(f"R² = {r2:.4f}")
if r2 < 0.2:
    print("➤ Weak explanatory power: Wind speed explains limited variance.")
elif r2 < 0.5:
    print("➤ Moderate explanatory power.")
else:
    print("➤ Strong explanatory power.")

# Optional plot
plt.figure()
plt.scatter(speed, onshore, alpha=0.3)
plt.plot(speed, reg.predict(X), color='red')
plt.title("Wind Speed vs Onshore Wind Generation")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Onshore Wind Generation (MW)")
plt.grid(True)
plt.tight_layout()
plt.show()
