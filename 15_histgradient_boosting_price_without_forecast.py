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
energy = energy.drop_duplicates(subset=["datetime"], keep="first")
weather_avg = weather.groupby("datetime").mean(numeric_only=True).reset_index()

df = pd.merge(energy, weather_avg, on="datetime", how="inner")
df = df.set_index("datetime").sort_index()

target = "price actual"

# Eksikleri doldur
df[target] = df[target].interpolate(method="time")

# ============================================================================
# 3. FEATURE ENGINEERING (FORECAST VERİLERİ YOK ❌)
# ============================================================================
# A) Takvim
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month

# B) GECİKMELİ FİYATLAR (Artık Modelin En Büyük Gücü Burası)
# Model artık "Grid Operatörü ne dedi?" diye bakamıyor, sadece "Geçmişte ne oldu?" diye bakabiliyor.
df["price_lag_24h"] = df[target].shift(24)    # Dün
df["price_lag_48h"] = df[target].shift(48)    # 2 Gün Önce (Yeni Ekledik)
df["price_lag_168h"] = df[target].shift(168)  # Geçen Hafta (Aynı Gün/Saat)

# C) Hareketli Ortalamalar (Trendi Yakalamak İçin)
df["price_roll_24h"] = df[target].shift(1).rolling(24).mean()

# Model Özellikleri (Forecast sütunları çıkarıldı)
features = ["temp", "humidity", "wind_speed",         # Hava Durumu
            "hour", "dayofweek", "month",             # Zaman
            "price_lag_24h", "price_lag_48h", "price_lag_168h", "price_roll_24h"] # Geçmiş Fiyatlar

# Boşlukları temizle
df_model = df[[target] + features].dropna()

# 4. MODEL EĞİTİMİ
test_size = 1000
X_train = df_model[features].iloc[:-test_size]
y_train = df_model[target].iloc[:-test_size]
X_test = df_model[features].iloc[-test_size:]
y_test = df_model[target].iloc[-test_size:]

print("Fiyat Modeli (Forecast Verisi Olmadan) Eğitiliyor...")
model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 5. SONUÇLAR
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)

print(f"\n=== FİYAT TAHMİN SONUÇLARI (SAF GEÇMİŞ VERİ İLE) ===")
print(f"R² Başarı Skoru: {r2:.4f}")
print(f"Ortalama Hata: {mae:.2f} EUR/MWh")

# Görselleştirme
plt.figure(figsize=(15, 6))
plt.plot(y_test.index, y_test, label="Gerçek Fiyat", color="black", alpha=0.7)
plt.plot(y_test.index, pred, label="Tahmin (Sadece Geçmiş Veri)", color="red", linestyle="--")
plt.title(f"Elektrik Fiyat Tahmini (Forecast Verisi YOK)\nR²={r2:.3f}")
plt.xlabel("Tarih")
plt.ylabel("Fiyat (EUR)")
plt.legend()
plt.show()