import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load datasets
energy = pd.read_csv("energy_dataset.csv")
weather = pd.read_csv("weather_features.csv")

# Parse datetime and align to hour
energy['time'] = pd.to_datetime(energy['time'], errors='coerce', utc=True)
weather['dt_iso'] = pd.to_datetime(weather['dt_iso'].str.slice(0, 19), errors='coerce', utc=True)

# Focus on Madrid (or adapt to all if needed)
weather_madrid = weather[weather['city_name'].str.strip() == 'Madrid'].drop_duplicates(subset='dt_iso')

# Round timestamps to the hour
energy['time_rounded'] = energy['time'].dt.round('h')
weather_madrid['dt_rounded'] = weather_madrid['dt_iso'].dt.round('h')

# Map weather condition to energy data
weather_dict = weather_madrid.set_index('dt_rounded')['weather_main'].str.lower().to_dict()
energy['weather_main'] = energy['time_rounded'].map(weather_dict)

# Drop missing values
df = energy[['generation solar', 'weather_main']].dropna()

# Bin solar generation into 3 quantile-based categories
df['solar_bin'] = pd.qcut(df['generation solar'], q=3, labels=['Low', 'Medium', 'High'])

# Build contingency table
contingency = pd.crosstab(df['weather_main'], df['solar_bin'])

# Chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency)

# Cramér’s V calculation
n = contingency.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

# Output results
print("=== Chi-squared Test ===")
print(f"Chi² = {chi2:.4f}, p = {p:.4f}")
if p < 0.05:
    print("➤ Reject H₀: Weather type significantly influences solar generation.")
else:
    print("➤ Fail to reject H₀: No significant influence detected.")

print("\n=== Cramér’s V ===")
print(f"Cramér’s V = {cramers_v:.4f}")
if cramers_v < 0.1:
    print("➤ Weak association strength.")
elif cramers_v < 0.3:
    print("➤ Moderate association strength.")
else:
    print("➤ Strong association strength.")
