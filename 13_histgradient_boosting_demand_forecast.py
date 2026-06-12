import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# 1. VERİ YÜKLEME VE TEMİZLİK
energy = pd.read_csv("energy_dataset.csv")
weather = pd.read_csv("weather_features.csv")

# Tarih formatlarını düzelt (UTC)
energy["datetime"] = pd.to_datetime(energy["time"], utc=True)
weather["datetime"] = pd.to_datetime(weather["dt_iso"], utc=True)

# Energy tablosundaki kopyaları temizle
energy = energy.drop_duplicates(subset=["datetime"], keep="first")

# Hava durumunu saatlik ortalamaya indir (Şehir bazlı veri varsa)
weather_avg = weather.groupby("datetime").agg({
    "temp": "mean",
    "humidity": "mean",
    "clouds_all": "mean",
    "wind_speed": "mean"
}).reset_index()

# Verileri birleştir
df = pd.merge(energy[["datetime", "total load actual"]], weather_avg, on="datetime", how="inner")
df = df.set_index("datetime").sort_index()

# Eksik verileri doldur (Interpolation)
target = "total load actual"
weather_features = ["temp", "humidity", "clouds_all", "wind_speed"]

df[target] = df[target].interpolate(method="time")
for f in weather_features:
    df[f] = df[f].interpolate(method="time")

# ==============================================================================
# 2. FEATURE ENGINEERING (BAŞARIYI GETİREN KISIM)
# ==============================================================================
# A) Takvim Özellikleri (İnsan davranışını yakalar)
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int) # Hafta sonu mu?

# B) Gecikmeli Özellikler (Lag Features - En önemli kısım!)
# Modelin "Dün bu saatte ne olmuştu?" diye bakmasını sağlar
df["load_lag_24h"] = df[target].shift(24) 
# Modelin "Geçen hafta bugün ne olmuştu?" diye bakmasını sağlar (Haftalık döngü)
df["load_lag_168h"] = df[target].shift(168)
# Son 24 saatin ortalama trendi
df["load_roll_mean_24h"] = df[target].shift(1).rolling(24).mean()

# Lag'lerden oluşan boşlukları temizle
features_to_use = weather_features + ["hour", "dayofweek", "month", "is_weekend", 
                                      "load_lag_24h", "load_lag_168h", "load_roll_mean_24h"]
df_model = df[[target] + features_to_use].dropna()

# 3. MODEL EĞİTİMİ (Train/Test Split)
# Son 6 ayı test için ayıralım
test_size = 24 * 180 
train = df_model.iloc[:-test_size]
test = df_model.iloc[-test_size:]

X_train = train[features_to_use]
y_train = train[target]
X_test = test[features_to_use]
y_test = test[target]

print(f"Eğitim seti: {len(X_train)} saat, Test seti: {len(X_test)} saat")

# HistGradientBoostingRegressor (LightGBM'in sklearn versiyonu - Çok Hızlı ve Güçlü)
model = HistGradientBoostingRegressor(max_iter=300, random_state=42, learning_rate=0.1)
model.fit(X_train, y_train)

# 4. DEĞERLENDİRME
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
print(f"R² Başarı Skoru: {r2:.4f}")

# Son 2 haftayı görselleştir
plt.figure(figsize=(14, 6))
zoom_idx = -24 * 14
plt.plot(y_test.index[zoom_idx:], y_test.iloc[zoom_idx:], label="Gerçek Talep", color='black', alpha=0.6)
plt.plot(y_test.index[zoom_idx:], pred[zoom_idx:], label="Model Tahmini", color='red', linestyle='--')
plt.title(f"Enerji Talebi Tahmini (R²={r2:.2f})")
plt.legend()
plt.show()