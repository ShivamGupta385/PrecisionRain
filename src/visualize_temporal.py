# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
from dask.diagnostics import ProgressBar

# Verify GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Using GPU: {physical_devices[0]}")
    except RuntimeError as e:
        print(e)

print("\n" + "="*40)
print("1. DATA LOADING & PREPARATION")
print("="*40)

file_path = 'data/India_Meteo_Combined_Final.nc'
ds = xr.open_dataset(file_path).chunk({'time': 500})

mask_lat = (ds.lat >= 23.08) & (ds.lat <= 30.24)
mask_lon = (ds.lon >= 69.45) & (ds.lon <= 78.32)
rajasthan_ds = ds.where(mask_lat & mask_lon, drop=True)

with ProgressBar():
    print("Crunching spatial data...")
    ds_monthly_rain = rajasthan_ds['rain'].resample(time='ME').sum()
    ds_monthly_temp = rajasthan_ds[['tmax', 'tmin']].resample(time='ME').mean()
    ds_monthly = xr.merge([ds_monthly_rain, ds_monthly_temp]).compute()

df = ds_monthly.to_dataframe().dropna().reset_index()
df.sort_values(by=['lat', 'lon', 'time'], inplace=True)

df['tmean'] = (df['tmax'] + df['tmin']) / 2.0
df['t_range'] = df['tmax'] - df['tmin']
df['rain_lag_1'] = df.groupby(['lat', 'lon'])['rain'].shift(1)
df['month_sin'] = np.sin(2 * np.pi * df['time'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['time'].dt.month / 12)
df.dropna(inplace=True)

features = ['rain', 'tmax', 'tmin', 'tmean', 't_range', 'rain_lag_1', 'month_sin', 'month_cos']
num_features = len(features)

scaler = RobustScaler()
df[features] = scaler.fit_transform(df[features])
df.set_index('time', inplace=True)

print("\n" + "="*40)
print("2. EXTRACTING SPATIAL SEQUENCES")
print("="*40)
SEQUENCE_LENGTH = 108 

def get_test_sequences(df, seq_length, feature_cols):
    X_test, X_geo_test, y_test, dates_test = [], [], [], []
    grouped_df = df.groupby(['lat', 'lon'])
    
    for (lat, lon), group in tqdm(grouped_df, desc="Processing Grid Points"):
        data = group[feature_cols].values
        dates = group.index.values
        if len(data) <= seq_length: continue
        
        loc_X, loc_y, loc_dates = [], [], []
        for i in range(len(data) - seq_length):
            loc_X.append(data[i : i + seq_length])
            loc_y.append(data[i + seq_length, 0]) 
            loc_dates.append(dates[i + seq_length])
            
        split_idx = int(len(loc_X) * 0.8)
        X_test.extend(loc_X[split_idx:])
        X_geo_test.extend([[lat, lon]] * (len(loc_X) - split_idx))
        y_test.extend(loc_y[split_idx:])
        dates_test.extend(loc_dates[split_idx:])
            
    return np.array(X_test), np.array(X_geo_test), np.array(y_test), np.array(dates_test)

X_seq_test, X_geo_test, y_test, target_dates_test = get_test_sequences(df, SEQUENCE_LENGTH, features)


print("\n" + "="*40)
print("3. PREDICTING & INVERSE TRANSFORMING (TRUE MM)")
print("="*40)

model = load_model("optimized_monsoon_model.keras")
predictions_scaled = model.predict([X_seq_test, X_geo_test], batch_size=2048).flatten()

# --- CONVERT SCALED NUMBERS BACK TO MILLIMETERS ---
dummy_pred = np.zeros((len(predictions_scaled), num_features))
dummy_pred[:, 0] = predictions_scaled
pred_mm = scaler.inverse_transform(dummy_pred)[:, 0]

dummy_actual = np.zeros((len(y_test), num_features))
dummy_actual[:, 0] = y_test
actual_mm = scaler.inverse_transform(dummy_actual)[:, 0]

test_df = pd.DataFrame({
    'date': pd.to_datetime(target_dates_test),
    'lat': X_geo_test[:, 0],
    'lon': X_geo_test[:, 1],
    'actual_mm': actual_mm,
    'predicted_mm': pred_mm
})

print("\n" + "="*40)
print("4. GENERATING FIGURE 5 LINE PLOTS")
print("="*40)

# Exact Coordinates derived from the paper's caption
target_locations = [
    {"letter": "(A)", "target_lat": 29.000, "target_lon": 73.000},
    {"letter": "(B)", "target_lat": 26.000, "target_lon": 74.833}, 
    {"letter": "(C)", "target_lat": 26.833, "target_lon": 75.000}, 
    {"letter": "(D)", "target_lat": 25.416, "target_lon": 75.833} 
]

fig, axes = plt.subplots(4, 1, figsize=(10, 18))
fig.tight_layout(pad=8.0)

available_lats = test_df['lat'].unique()
available_lons = test_df['lon'].unique()

for idx, loc in enumerate(target_locations):
    # Find the closest matching grid point in our NetCDF data
    nearest_lat = available_lats[np.argmin(np.abs(available_lats - loc["target_lat"]))]
    nearest_lon = available_lons[np.argmin(np.abs(available_lons - loc["target_lon"]))]
    
    # Filter the dataframe for this specific location and sort chronologically
    loc_df = test_df[(test_df['lat'] == nearest_lat) & (test_df['lon'] == nearest_lon)].copy()
    loc_df.sort_values('date', inplace=True)
    
    # (Notice we REMOVED the month filter here so the winter months stay flat at 0mm!)

    ax = axes[idx]
    
    # Plotting styles matched to Figure 5
    ax.plot(loc_df['date'], loc_df['actual_mm'], label='Actual rainfall', color='#2c7bb6', linewidth=1.5)
    ax.plot(loc_df['date'], loc_df['predicted_mm'], label='Predicted rainfall', color='#fdae61', linestyle='-.', linewidth=1.5)
    
    # Formatting the subplot
    ax.set_title(loc['letter'], y=-0.35, fontsize=14, fontweight='bold')
    ax.set_ylabel('Rainfall (mm)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    
    # Put the legend inside the top right corner
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    
    # --- NEW X-AXIS FORMATTING ---
    # Instead of YearLocator, we force it to use the actual dates in your dataframe.
    # We use [::4] to only print every 4th month so the text doesn't overlap and turn into a black blob.
    ax.set_xticks(loc_df['date'][::4])
    ax.set_xticklabels(loc_df['date'].dt.strftime('%b-%Y')[::4], rotation=90, fontsize=8)

output_filename = "images/temporal_trends.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nSUCCESS: Line plots saved as '{output_filename}' in your directory!")