import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, shapiro, levene  # <-- Levene eklendi
import numpy as np

# 1. Load and preprocess data
# Dosya yollarının doğru olduğundan emin olun
energy = pd.read_csv("energy_dataset.csv")
energy['time'] = pd.to_datetime(energy['time'], errors='coerce', utc=True)
energy['time_rounded'] = energy['time'].dt.round('h')

weather = pd.read_csv("weather_features.csv")
weather['dt_iso'] = pd.to_datetime(weather['dt_iso'].str.slice(0, 19), errors='coerce', utc=True)
weather['dt_rounded'] = weather['dt_iso'].dt.round('h')

# Convert Kelvin to Celsius
weather['temp'] = weather['temp'] - 273.15

# Average temperature across all Spanish cities per hour
avg_weather = weather.groupby('dt_rounded')['temp'].mean().reset_index().rename(columns={'temp': 'avg_temp'})

# Merge datasets
energy_temp = pd.merge(energy, avg_weather, left_on='time_rounded', right_on='dt_rounded', how='inner')
energy_temp = energy_temp.dropna(subset=['total load actual'])

# 2. Temperature bands
low_temp = energy_temp[energy_temp['avg_temp'] < 10]
high_temp = energy_temp[energy_temp['avg_temp'] > 25]

print(f"Low temp records: {len(low_temp)} | High temp records: {len(high_temp)}")

if len(low_temp) < 30 or len(high_temp) < 30:
    print("❌ Not enough samples to draw 30 each.")
    exit()

# 3. Sampling
low_sample = low_temp['total load actual'].sample(n=30, random_state=42)
high_sample = high_temp['total load actual'].sample(n=30, random_state=42)

# 4. Descriptive Statistics
print("\n=== Descriptive Stats ===")
print(f"Low Temp: mean={low_sample.mean():.2f}, std={low_sample.std():.2f}")
print(f"High Temp: mean={high_sample.mean():.2f}, std={high_sample.std():.2f}")

# 5. Normality Check (Shapiro-Wilk)
print("\n=== Shapiro-Wilk Normality Test ===")
print(f"Low Temp: p = {shapiro(low_sample).pvalue:.4f}")
print(f"High Temp: p = {shapiro(high_sample).pvalue:.4f}")

# 6. Variance Homogeneity Check (Levene's Test)
# BU BÖLÜM EKLENDİ
print("\n=== Levene's Test for Equality of Variances ===")
stat_levene, p_levene = levene(low_sample, high_sample)
print(f"Levene Stat: {stat_levene:.4f}, p-value: {p_levene:.4f}")

if p_levene < 0.05:
    print("➤ Reject H₀: Variances are UNEQUAL.")
    print("➤ Justification: Proceed with Welch's T-test (equal_var=False).")
else:
    print("➤ Fail to reject H₀: Variances are equal.")
    print("➤ Note: Standard T-test could be used, but Welch's is safer.")

# 7. T-Test (Welch's because equal_var=False)
print("\n=== Independent Samples T-Test (Welch's) ===")
t_stat, p_val = ttest_ind(low_sample, high_sample, equal_var=False)
print(f"T-Statistic: {t_stat:.4f}, p-value: {p_val:.4f}")

if p_val < 0.05:
    print("➤ Conclusion: Reject H₀. Significant difference in energy demand between temp bands.")
else:
    print("➤ Conclusion: Fail to reject H₀. No significant difference.")

# 8. Visualizations
sns.set(style="whitegrid")

# Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=[low_sample, high_sample], palette=['skyblue', 'lightcoral'])
plt.xticks([0, 1], ['Low Temp (<10°C)', 'High Temp (>25°C)'])
plt.title("Boxplot: Energy Demand vs Temperature Extremes")
plt.ylabel("Total Load (MW)")
plt.savefig("t_boxplot_temp_demand.png")
plt.show()

# Violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(data=[low_sample, high_sample], palette=['skyblue', 'lightcoral'])
plt.xticks([0, 1], ['Low Temp (<10°C)', 'High Temp (>25°C)'])
plt.title("Violin Plot: Demand Distribution by Temperature")
plt.ylabel("Total Load (MW)")
plt.savefig("t_violin_temp_demand.png")
plt.show()

# Histogram + KDE
plt.figure(figsize=(10, 6))
sns.histplot(low_sample, kde=True, color='skyblue', label='Low Temp (<10°C)', stat='density', bins=15)
sns.histplot(high_sample, kde=True, color='lightcoral', label='High Temp (>25°C)', stat='density', bins=15)
plt.axvline(low_sample.mean(), color='blue', linestyle='--', label=f"Low Mean: {low_sample.mean():.0f}")
plt.axvline(high_sample.mean(), color='red', linestyle='--', label=f"High Mean: {high_sample.mean():.0f}")
plt.title("Distribution of Energy Demand Samples")
plt.xlabel("Total Load (MW)")
plt.legend()
plt.tight_layout()
plt.savefig("t_distribution_temp_demand.png")
plt.show()