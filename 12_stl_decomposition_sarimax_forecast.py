import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL # Daha esnek decomposition
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----------------------------
# 0) LOAD + MERGE + CLEAN
# ----------------------------
energy = pd.read_csv("energy_dataset.csv")
weather = pd.read_csv("weather_features.csv")

energy["time"] = pd.to_datetime(energy["time"], utc=True)
weather["dt_iso"] = pd.to_datetime(weather["dt_iso"], utc=True)

energy = energy.rename(columns={"time": "datetime"})
weather = weather.rename(columns={"dt_iso": "datetime"})

# Şehir bazlı hava durumunu ortalamaya alalım (Ulusal yük tahmini için)
weather_avg = weather.groupby("datetime").agg({
    "temp": "mean",
    "humidity": "mean",
    "clouds_all": "mean",
    "wind_speed": "mean"
}).reset_index()

df = pd.merge(energy[["datetime", "total load actual"]], weather_avg, on="datetime", how="inner")
df = df.dropna().sort_values("datetime")
df = df.set_index("datetime").asfreq("H")
df = df.interpolate(method="time")

target = "total load actual"
features = ["temp", "humidity", "clouds_all", "wind_speed"]

# ============================================================
# 1) ADVANCED DECOMPOSITION (STL)
# ============================================================
print("\n[1] Decomposition Analizi Yapılıyor...")
# STL, mevsimselliğin zamanla değişmesine izin verir
stl = STL(df[target], period=24, robust=True)
res = stl.fit()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
res.observed.plot(ax=ax1, title="Gözlemlenen")
res.trend.plot(ax=ax2, title="Trend (Uzun Vadeli Eğilim)")
res.seasonal.plot(ax=ax3, title="Mevsimsellik (Günlük Döngü)")
res.resid.plot(ax=ax4, title="Artıklar (Gürültü)")
plt.tight_layout()
plt.show()

# Yorumlama
resid_std = res.resid.std()
target_std = df[target].std()
print(f"-> Gürültü/Sinyal Oranı: {resid_std/target_std:.4f}")

# ============================================================
# 2) FEATURE ENGINEERING (Lagged & Rolling)
# ============================================================
print("\n[2] Özellik Mühendisliği (Lags & Rolling) Uygulanıyor...")
df_eng = df.copy()
for f in features:
    df_eng[f"{f}_lag_6h"] = df_eng[f].shift(6)
    df_eng[f"{f}_roll_6h"] = df_eng[f].rolling(window=6).mean()

df_eng = df_eng.dropna()

# ============================================================
# 3) FORECASTING: SARIMAX (Hava Durumu Katkılı)
# ============================================================
TRAIN_SIZE = int(len(df_eng) * 0.95) # Son %5 test
train, test = df_eng.iloc[:TRAIN_SIZE], df_eng.iloc[TRAIN_SIZE:]

# Hızlı sonuç için son 2 ayın verisini kullanalım
train_recent = train.last("60D")

print("\n[3] SARIMAX Tahminleme Başlatıldı...")
model_exog = SARIMAX(
    train_recent[target],
    exog=train_recent[[f"{f}_lag_6h" for f in features]],
    order=(1, 1, 1),
    seasonal_order=(0, 1, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False
)

fit_exog = model_exog.fit(disp=False, maxiter=50)
forecast_steps = len(test)
pred = fit_exog.get_forecast(steps=forecast_steps, exog=test[[f"{f}_lag_6h" for f in features]]).predicted_mean

# Metrikler
mae = mean_absolute_error(test[target], pred)
rmse = np.sqrt(mean_squared_error(test[target], pred))
r2 = r2_score(test[target], pred)

# ============================================================
# 4) OTOMATİK YORUMLAMA (Terminal Çıktısı)
# ============================================================
print("\n" + "="*50)
print("ANALİZ VE MODEL PERFORMANS RAPORU")
print("="*50)

# 1. Decomposition Yorumu
print(f"1. DECOMPOSITION:")
if resid_std / target_std < 0.2:
    print("   ✅ BAŞARILI: Veride çok güçlü bir yapı var (Trend/Mevsimsellik hakim).")
else:
    print("   ⚠️ UYARI: Veride gürültü yüksek, tahmin zorlaşabilir.")

# 2. SARIMAX Performans Yorumu
print(f"\n2. SARIMAX MODELİ (Hava Durumu Destekli):")
print(f"   - MAE: {mae:.2f} MW")
print(f"   - RMSE: {rmse:.2f} MW")
print(f"   - R² Skoru: {r2:.4f}")

if r2 > 0.8:
    print("   ✅ MÜKEMMEL: Model enerji talebindeki değişimin %80'inden fazlasını açıklıyor.")
elif r2 > 0.5:
    print("   OK: Model genel yönü biliyor ama sapmalar var.")
else:
    print("   ❌ BAŞARISIZ: Model verideki deseni yakalayamadı.")

# 3. Hava Durumu Etkisi Yorumu
weather_corr = df_eng[[target, 'temp', 'humidity']].corr().iloc[0, 1]
print(f"\n3. HAVA DURUMU ETKİSİ:")
print(f"   - Sıcaklık/Talep Korelasyonu: {weather_corr:.2f}")
if abs(weather_corr) > 0.4:
    print("   ✅ ANLAMLI: Hava durumu verileri tahmin için kritik öneme sahip.")
else:
    print("   ℹ️ DÜŞÜK: Bu bölgede talep hava durumundan ziyade saate/güne bağlı.")

# Görselleştirme
plt.figure(figsize=(12, 5))
plt.plot(test.index, test[target], label="Gerçek Talep", color='black', alpha=0.7)
plt.plot(test.index, pred, label="SARIMAX Tahmini", color='red', linestyle='--')
plt.title("Enerji Talebi Tahmini: Gerçek vs Tahmin")
plt.legend()
plt.show()

print("\n" + "="*50)