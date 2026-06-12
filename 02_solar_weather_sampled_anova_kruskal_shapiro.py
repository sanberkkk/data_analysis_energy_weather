import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, kruskal, shapiro
import numpy as np

# Load datasets
energy = pd.read_csv("energy_dataset.csv")
energy['time'] = pd.to_datetime(energy['time'], errors='coerce', utc=True)

weather = pd.read_csv("weather_features.csv")
weather['dt_iso'] = pd.to_datetime(weather['dt_iso'].str.slice(0, 19), errors='coerce', utc=True)
weather_madrid = weather[weather['city_name'].str.strip() == 'Madrid'].drop_duplicates(subset='dt_iso')

# Time alignment
energy['time_rounded'] = energy['time'].dt.round('H')
weather_madrid['dt_rounded'] = weather_madrid['dt_iso'].dt.round('H')

# Map weather condition directly
weather_labels = weather_madrid.set_index('dt_rounded')['weather_main'].str.lower().to_dict()
energy['weather_main'] = energy['time_rounded'].map(weather_labels)

# Drop missing
data = energy.dropna(subset=['generation solar', 'weather_main'])

# Sample 30 entries per weather condition (if available)
np.random.seed(42)
sampled_data = data.groupby('weather_main').apply(lambda x: x.sample(n=30) if len(x) >= 30 else None).dropna(subset=['generation solar']).reset_index(drop=True)

# Group for stats and tests
grouped = sampled_data.groupby('weather_main')['generation solar']

# Descriptive stats
stats = grouped.agg(['count', 'mean', 'median', 'std', 'var'])
print("\nDescriptive Statistics (Sampled Solar Generation by Weather Type):")
print(stats)

# Shapiro-Wilk Test
print("\nShapiro-Wilk Normality Test (Sampled):")
for name, group in grouped:
    if len(group) >= 3:
        stat, p = shapiro(group)
        print(f"{name.capitalize()}: p = {p:.4f}")
    else:
        print(f"{name.capitalize()}: Not enough data")

# Hypothesis Tests
samples = [group for _, group in grouped if len(group) >= 3]

anova_stat, anova_p = f_oneway(*samples)
print(f"\nANOVA Test (Sampled): F = {anova_stat:.4f}, p = {anova_p:.4f}")
print("➤", "Reject H₀: Significant difference." if anova_p < 0.05 else "Fail to reject H₀.")

kruskal_stat, kruskal_p = kruskal(*samples)
print(f"Kruskal-Wallis Test (Sampled): H = {kruskal_stat:.4f}, p = {kruskal_p:.4f}")
print("➤", "Reject H₀: Significant difference in medians." if kruskal_p < 0.05 else "Fail to reject H₀.")

# Violin Plot
plt.figure()
sns.violinplot(x='weather_main', y='generation solar', data=sampled_data)
plt.xticks(rotation=45)
plt.title("Violin Plot: Sampled Solar Generation by Weather")
plt.ylabel("Solar Generation (MW)")
plt.tight_layout()
plt.savefig("violin_by_weather_type.png")
plt.show()

# Boxplot
plt.figure()
sns.boxplot(x='weather_main', y='generation solar', data=sampled_data)
plt.xticks(rotation=45)
plt.title("Boxplot: Sampled Solar Generation by Weather")
plt.ylabel("Solar Generation (MW)")
plt.tight_layout()
plt.savefig("boxplot_by_weather_type.png")
plt.show()

# Barplot Mean ± SEM
means = grouped.mean()
sems = grouped.sem()
plt.figure()
means.plot(kind='bar', yerr=sems, capsize=5, color='cornflowerblue', edgecolor='black')
plt.ylabel("Mean Solar Generation (MW)")
plt.title("Mean ± SEM of Sampled Solar Generation by Weather Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("barplot_mean_sem_by_weather.png")
plt.show()
