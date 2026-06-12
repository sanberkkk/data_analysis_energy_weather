import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. VERİ YÜKLEME
energy = pd.read_csv("energy_dataset.csv")
weather = pd.read_csv("weather_features.csv")

# Tarih Ayarları
energy["datetime"] = pd.to_datetime(energy["time"], utc=True)
weather["datetime"] = pd.to_datetime(weather["dt_iso"], utc=True)

# 2. TEMİZLİK & BİRLEŞTİRME
# Tekrarlayan saatleri temizle
energy = energy.drop_duplicates(subset=["datetime"], keep="first")

# Hava durumunu ortalamaya al
weather_avg = weather.groupby("datetime").mean(numeric_only=True).reset_index()

# Birleştir
df = pd.merge(energy, weather_avg, on="datetime", how="inner")
df = df.set_index("datetime").sort_index()

# Hedef: GERÇEK FİYAT
target = "price actual"

# Eksikleri doldur (Interpolation)
df[target] = df[target].interpolate(method="time")

# ============================================================================
# 3. FEATURE ENGINEERING (FİYAT İÇİN ÖZEL)
# ============================================================================
# A) Takvim
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month

# B) PİYASA VERİLERİ (Çok Önemli!)
# Fiyatı belirleyen en büyük etken, sistemin o gün için beklediği yük ve üretimdir.
# Bu sütunlar verisetinde mevcut ve geleceğe dönük kullanılabilir ("Forecast" oldukları için).
market_features = [
    "total load forecast",             # Beklenen Talep
    "forecast solar day ahead",        # Beklenen Güneş
    "forecast wind onshore day ahead"  # Beklenen Rüzgar
]

# C) GECİKMELİ FİYATLAR (Lags)
df["price_lag_24h"] = df[target].shift(24)    # Dün bu saatte fiyat neydi?
df["price_lag_168h"] = df[target].shift(168)  # Geçen hafta aynı saat?

# Model için kullanılacak tüm sütunlar
features = ["temp", "humidity", "wind_speed", "hour", "dayofweek", "month",
            "price_lag_24h", "price_lag_168h"] + market_features

# Boşlukları (Lag'lerden oluşan) temizle
df_model = df[[target] + features].dropna()

# 4. MODEL EĞİTİMİ (Gradient Boosting)
# Son 1000 saati test için ayıralım
test_size = 1000
X_train = df_model[features].iloc[:-test_size]
y_train = df_model[target].iloc[:-test_size]
X_test = df_model[features].iloc[-test_size:]
y_test = df_model[target].iloc[-test_size:]

print("Fiyat Modeli Eğitiliyor...")
model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 5. SONUÇLAR
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)

print(f"\n=== FİYAT TAHMİN SONUÇLARI ===")
print(f"R² Başarı Skoru: {r2:.4f} (1.00 en iyi)")
print(f"Ortalama Hata: {mae:.2f} EUR/MWh")

# Görselleştirme
plt.figure(figsize=(15, 6))
plt.plot(y_test.index, y_test, label="Gerçek Fiyat", color="black", alpha=0.7)
plt.plot(y_test.index, pred, label="Tahmin Edilen Fiyat", color="green", linestyle="--")
plt.title(f"Elektrik Fiyat Tahmini (Forecast Verileri Dahil)\nR²={r2:.3f}")
plt.xlabel("Tarih")
plt.ylabel("Fiyat (EUR)")
plt.legend()
plt.show()