import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. VERİ YÜKLEME
energy = pd.read_csv("energy_dataset.csv")
weather = pd.read_csv("weather_features.csv")

energy["datetime"] = pd.to_datetime(energy["time"], utc=True)
weather["datetime"] = pd.to_datetime(weather["dt_iso"], utc=True)

# 2. TOPLAM ÜRETİMİ HESAPLAMA (Total Generation)
# "generation" ile başlayan tüm sütunları bul ve topla
gen_cols = [col for col in energy.columns if col.startswith("generation")]
# (Boş olan 'hydro pumped storage aggregated' sütununu çıkaralım)
if "generation hydro pumped storage aggregated" in gen_cols:
    gen_cols.remove("generation hydro pumped storage aggregated")

energy["total_generation"] = energy[gen_cols].sum(axis=1)

# Temizlik & Birleştirme
energy = energy.drop_duplicates(subset=["datetime"], keep="first")
weather_avg = weather.groupby("datetime").mean(numeric_only=True).reset_index()

df = pd.merge(energy, weather_avg, on="datetime", how="inner")
df = df.set_index("datetime").sort_index()

target = "total_generation"

# 3. FEATURE ENGINEERING
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# Gecikmeli Veriler (Lags)
df["gen_lag_24h"] = df[target].shift(24)      # Dün ne kadar üretildi?
df["gen_lag_168h"] = df[target].shift(168)    # Geçen hafta?
df["gen_roll_mean"] = df[target].shift(1).rolling(24).mean() # Son 24 saat ortalaması

# Ekstra Bilgi: Talep Tahmini (Üretim talebi takip eder)
df["load_forecast"] = df["total load forecast"]

features = ["temp", "humidity", "wind_speed", "hour", "dayofweek", 
            "gen_lag_24h", "gen_lag_168h", "gen_roll_mean", "load_forecast"]

df_model = df[[target] + features].dropna()

# 4. MODEL EĞİTİMİ
test_size = 1000
X_train = df_model[features].iloc[:-test_size]
y_train = df_model[target].iloc[:-test_size]
X_test = df_model[features].iloc[-test_size:]
y_test = df_model[target].iloc[-test_size:]

print("Üretim Modeli Eğitiliyor...")
model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 5. SONUÇLAR
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)

print(f"\n=== TOPLAM ÜRETİM TAHMİN SONUÇLARI ===")
print(f"R² Başarı Skoru: {r2:.4f}")
print(f"Ortalama Hata: {mae:.2f} MW")

# Görselleştirme
plt.figure(figsize=(15, 6))
plt.plot(y_test.index, y_test, label="Gerçek Üretim", color="navy", alpha=0.6)
plt.plot(y_test.index, pred, label="Model Tahmini", color="orange", linestyle="--")
plt.title(f"Toplam Enerji Üretimi Tahmini\nR²={r2:.3f}")
plt.xlabel("Tarih")
plt.ylabel("Üretim (MW)")
plt.legend()
plt.show()