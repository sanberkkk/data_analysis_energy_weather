import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
energy = pd.read_csv("energy_dataset.csv")
energy['time'] = pd.to_datetime(energy['time'], errors='coerce', utc=True)
energy = energy.dropna(subset=['total load actual'])

# Assign seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

energy['season'] = energy['time'].dt.month.map(get_season)
energy['year'] = energy['time'].dt.year

# Average generation by season and year
seasonal_avg = energy.groupby(['year', 'season'])['total load actual'].mean().unstack()

# Drop years with missing data for any season
seasonal_avg = seasonal_avg.dropna()

# Friedman test
stat, p = friedmanchisquare(seasonal_avg['Winter'], seasonal_avg['Spring'],
                            seasonal_avg['Summer'], seasonal_avg['Autumn'])

# Print results
print("Descriptive Statistics by Season and Year:")
print(seasonal_avg.describe().T)

print(f"\nFriedman Test Results:\nStatistic = {stat:.4f}, p-value = {p:.4f}")
if p < 0.05:
    print("➤ Reject H₀: Energy generation differs significantly across seasons.")
else:
    print("➤ Fail to reject H₀: No significant seasonal variation detected.")

# Plotting
plt.figure(figsize=(8, 5))
seasonal_avg.plot(marker='o')
plt.title("Average Energy Generation by Season (per Year)")
plt.ylabel("Average Total Load Actual (MW)")
plt.xlabel("Year")
plt.grid(True)
plt.tight_layout()
plt.savefig("seasonal_generation_friedman.png")
plt.show()
