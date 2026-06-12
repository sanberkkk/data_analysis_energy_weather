import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene, bartlett, ttest_ind

# Load and filter data
df = pd.read_csv("spain_energy_weather_hourly.csv")
df = df[['time', 'temp_avg', 'total load actual']].dropna()

# Create temperature bands
df['temp_band'] = pd.cut(df['temp_avg'],
                         bins=[-float('inf'), 283.15, 298.15, float('inf')],
                         labels=['Low (<10°C)', 'Moderate (10-25°C)', 'High (>25°C)'])

# Group demand data by temperature bands
low = df[df['temp_band'] == 'Low (<10°C)']['total load actual']
high = df[df['temp_band'] == 'High (>25°C)']['total load actual']

# Calculate and print standard deviations
print("Standard Deviation by Temperature Band:")
print(df.groupby('temp_band')['total load actual'].std())

# Levene's Test (variance equality)
l_stat, l_p = levene(low, high)
print(f"\nLevene’s Test: W = {l_stat:.4f}, p = {l_p:.4f}")
if l_p < 0.05:
    print("➤ Reject H₀: Variance differs significantly.")
else:
    print("➤ Fail to reject H₀: No significant variance difference.")

# Bartlett's Test
b_stat, b_p = bartlett(low, high)
print(f"Bartlett’s Test: T = {b_stat:.4f}, p = {b_p:.4f}")
if b_p < 0.05:
    print("➤ Reject H₀: Variance differs significantly.")
else:
    print("➤ Fail to reject H₀: No significant variance difference.")

# T-test on standard deviations (not classical use, but for illustration)
t_stat, t_p = ttest_ind(low, high, equal_var=False)
print(f"\nT-test: t = {t_stat:.4f}, p = {t_p:.4f}")
if t_p < 0.05:
    print("➤ Reject H₀: Mean demand differs between low and high temp.")
else:
    print("➤ Fail to reject H₀: No significant mean difference.")

# Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='temp_band', y='total load actual', data=df, palette='coolwarm')
plt.title("Energy Demand by Temperature Band")
plt.ylabel("Total Load Actual (MW)")
plt.xlabel("Temperature Band")
plt.tight_layout()
plt.savefig("demand_by_temp_band_boxplot.png")
plt.show()
