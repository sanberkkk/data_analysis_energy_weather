import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("spain_energy_weather_hourly.csv")
df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)

# Correct weather columns based on dataset
weather_cols = ['temp_avg', 'humidity_avg', 'wind_speed_avg', 'clouds_avg']
demand_col = 'total load actual'

# Check existence
missing = [col for col in weather_cols + [demand_col] if col not in df.columns]
if missing:
    raise KeyError(f"Missing columns in dataset: {missing}")

# Drop NA
df_clean = df[['time', demand_col] + weather_cols].dropna()

# Normalize weather features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[weather_cols])

# KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
df_clean['weather_cluster'] = kmeans.fit_predict(X_scaled)

# Summary: demand profile by cluster
summary = df_clean.groupby('weather_cluster')[demand_col].agg(['mean', 'std', 'count'])
print("\n📊 Demand Profile by Weather Cluster:")
print(summary)

# Boxplot of demand by cluster
plt.figure(figsize=(8, 6))
sns.boxplot(x='weather_cluster', y=demand_col, data=df_clean, palette='Set2')
plt.title("Energy Demand by Weather Cluster")
plt.xlabel("Cluster")
plt.ylabel("Total Load Actual (MW)")
plt.tight_layout()
plt.savefig("cluster_demand_boxplot.png")
plt.show()

# Optional: Weather feature distribution
sample_df = df_clean.sample(n=1000, random_state=42)
sns.pairplot(sample_df, hue='weather_cluster', vars=weather_cols, palette='Set2')
plt.suptitle("Weather Clusters and Feature Distributions", y=1.02)
plt.tight_layout()
plt.savefig("cluster_weather_pairplot.png")
plt.show()
