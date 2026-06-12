import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, kruskal, shapiro
import numpy as np

# 1. Load and preprocess data
energy = pd.read_csv("energy_dataset.csv")
energy['time'] = pd.to_datetime(energy['time'], errors='coerce', utc=True)

weather = pd.read_csv("weather_features.csv")
weather['dt_iso'] = pd.to_datetime(weather['dt_iso'].str.slice(0, 19), errors='coerce', utc=True)
weather_madrid = weather[weather['city_name'].str.strip() == 'Madrid'].drop_duplicates(subset='dt_iso')

# Round to hourly for alignment
energy['time_rounded'] = energy['time'].dt.round('H')
weather_madrid['dt_rounded'] = weather_madrid['dt_iso'].dt.round('H')

# Build a weather label dictionary
weather_labels = weather_madrid.set_index('dt_rounded')['weather_main'].str.lower().to_dict()
energy['weather_main'] = energy['time_rounded'].map(weather_labels)

# Map weather to categories
def categorize_weather(w):
    if w == 'clear': return 'Sunny'
    elif w == 'clouds': return 'Cloudy'
    elif w == 'rain': return 'Rainy'
    else: return 'Other'

energy['Weather Condition'] = energy['weather_main'].map(categorize_weather)
data = energy.dropna(subset=['generation solar', 'Weather Condition'])

# 2. Descriptive Statistics
grouped = data.groupby('Weather Condition')['generation solar']
stats = grouped.agg(['count', 'mean', 'median', 'std', 'var'])
print("\nDescriptive Statistics (Solar Generation by Weather Condition):")
print(stats)

# 3. Shapiro-Wilk Normality Test
print("\nShapiro-Wilk Normality Test Results:")
for name, group in grouped:
    sample = group.sample(min(len(group), 500), random_state=42)
    if len(sample) >= 3:
        stat, p = shapiro(sample)
        print(f"{name}: p = {p:.4f}")
    else:
        print(f"{name}: Not enough data")

# 4. Hypothesis Testing
groups = [group for _, group in grouped if len(group) >= 3]

# ANOVA
anova_stat, anova_p = f_oneway(*groups)
print(f"\nANOVA Test: F = {anova_stat:.4f}, p = {anova_p:.4f}")
if anova_p < 0.05:
    print("➤ Reject H₀: Solar generation differs by weather condition.")
else:
    print("➤ Fail to reject H₀: No significant difference detected.")

# Kruskal-Wallis
kruskal_stat, kruskal_p = kruskal(*groups)
print(f"\nKruskal-Wallis Test: H = {kruskal_stat:.4f}, p = {kruskal_p:.4f}")
if kruskal_p < 0.05:
    print("➤ Reject H₀: Significant difference in medians.")
else:
    print("➤ Fail to reject H₀: No significant difference detected.")

# 5. Visualizations

# Violin Plot
plt.figure()
sns.violinplot(x='Weather Condition', y='generation solar', data=data,
               order=["Sunny", "Cloudy", "Rainy", "Other"],
               palette=['gold','gray','dodgerblue','lightgreen'])
plt.title("Violin Plot: Solar Generation Distribution")
plt.ylabel("Solar Generation (MW)")
plt.savefig("violin_solar_weather.png")
plt.show()

# Boxplot
plt.figure()
sns.boxplot(x='Weather Condition', y='generation solar', data=data,
            order=["Sunny", "Cloudy", "Rainy", "Other"],
            palette=['gold','gray','dodgerblue','lightgreen'])
plt.title("Boxplot: Solar Generation vs Weather Condition")
plt.ylabel("Solar Generation (MW)")
plt.savefig("boxplot_solar_weather.png")
plt.show()

# Bar Plot of Means + SEM
means = grouped.mean()
sems = grouped.sem()
plt.figure()
means.plot(kind='bar', yerr=sems, capsize=5, color='skyblue', edgecolor='black')
plt.ylabel("Mean Solar Generation (MW)")
plt.title("Mean ± SEM of Solar Generation by Weather")
plt.tight_layout()
plt.savefig("barplot_mean_sem_solar.png")
plt.show()
