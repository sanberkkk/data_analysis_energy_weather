import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, friedmanchisquare, shapiro
import numpy as np

# Load energy data
energy = pd.read_csv("energy_dataset.csv")
energy['time'] = pd.to_datetime(energy['time'], errors='coerce', utc=True)

# Extract month and map to season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

energy['month'] = energy['time'].dt.month
energy['season'] = energy['month'].apply(get_season)

# Total generation (could be renewable, fossil, etc. — use 'total load actual' for demand)
target_column = 'total load actual'
data = energy.dropna(subset=[target_column, 'season'])

# Group by season
season_groups = data.groupby('season')[target_column]

# Descriptive stats
stats = season_groups.agg(['count', 'mean', 'median', 'std', 'var'])
print("\nDescriptive Statistics by Season:")
print(stats)

# ANOVA
samples = [group for _, group in season_groups]
anova_stat, anova_p = f_oneway(*samples)
print(f"\nOne-way ANOVA: F = {anova_stat:.4f}, p = {anova_p:.4f}")
if anova_p < 0.05:
    print("➤ Reject H₀: Energy generation differs across seasons.")
else:
    print("➤ Fail to reject H₀: No significant seasonal difference.")

# Friedman test — using monthly means as repeated measures
monthly = data.groupby(['month', 'season'])[target_column].mean().reset_index()
wide_format = monthly.pivot(index='month', columns='season', values=target_column)
if wide_format.dropna().shape[0] >= 3:
    friedman_stat, friedman_p = friedmanchisquare(
        wide_format['Winter'], wide_format['Spring'], wide_format['Summer'], wide_format['Autumn']
    )
    print(f"\nFriedman Test: χ² = {friedman_stat:.4f}, p = {friedman_p:.4f}")
    if friedman_p < 0.05:
        print("➤ Reject H₀: Seasonal effect detected (repeated measures).")
    else:
        print("➤ Fail to reject H₀.")
else:
    print("❗ Not enough data for Friedman test.")

# Plot: Boxplot
plt.figure()
sns.boxplot(x='season', y=target_column, data=data, order=['Winter', 'Spring', 'Summer', 'Autumn'],
            palette='pastel')
plt.title("Energy Demand by Season")
plt.ylabel("Total Load Actual (MW)")
plt.savefig("season_boxplot.png")
plt.show()

# Plot: Barplot with Error Bars
means = season_groups.mean()
sems = season_groups.sem()
plt.figure()
means.loc[['Winter','Spring','Summer','Autumn']].plot(kind='bar', yerr=sems, capsize=5, color='skyblue')
plt.ylabel("Mean Energy Demand (MW)")
plt.title("Mean ± SEM of Energy Demand by Season")
plt.tight_layout()
plt.savefig("season_barplot_sem.png")
plt.show()
