# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import xarray as xr
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model  # <-- NEW: Only import load_model
from tqdm import tqdm
from dask.diagnostics import ProgressBar

print("\n" + "="*40)
print("1. DATA LOADING & VECTORIZED PREPROCESSING")
print("="*40)

file_path = 'data/India_Meteo_Combined_Final.nc'
ds = xr.open_dataset(file_path).chunk({'time': 500})

mask_lat = (ds.lat >= 23.08) & (ds.lat <= 30.24)
mask_lon = (ds.lon >= 69.45) & (ds.lon <= 78.32)
rajasthan_ds = ds.where(mask_lat & mask_lon, drop=True)

with ProgressBar():
    print("Crunching spatial data into memory:")
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
print("2. SEQUENCE GENERATION (TEST DATA ONLY)")
print("="*40)
SEQUENCE_LENGTH = 108 

def get_test_sequences(df, seq_length, feature_cols):
    X_test, X_geo_test, y_test, dates_test = [], [], [], []
    grouped_df = df.groupby(['lat', 'lon'])
    
    for (lat, lon), group in tqdm(grouped_df, desc="Processing Locations"):
        data = group[feature_cols].values
        dates = group.index.values
        
        if len(data) <= seq_length: continue
        
        loc_X, loc_y, loc_dates = [], [], []
        for i in range(len(data) - seq_length):
            loc_X.append(data[i : i + seq_length])
            loc_y.append(data[i + seq_length, 0]) 
            loc_dates.append(dates[i + seq_length])
            
        split_idx = int(len(loc_X) * 0.8)
        
        # We ONLY need the test data for evaluation!
        X_test.extend(loc_X[split_idx:])
        X_geo_test.extend([[lat, lon]] * (len(loc_X) - split_idx))
        y_test.extend(loc_y[split_idx:])
        dates_test.extend(loc_dates[split_idx:])
            
    return np.array(X_test), np.array(X_geo_test), np.array(y_test), np.array(dates_test)

X_seq_test, X_geo_test, y_test, target_dates_test = get_test_sequences(df, SEQUENCE_LENGTH, features)


print("\n" + "="*40)
print("3. LOADING PRE-TRAINED MODEL (NO TRAINING!)")
print("="*40)

# Load the saved model in 1 second
model = load_model("optimized_monsoon_model.keras")
print("Successfully loaded 'optimized_monsoon_model.keras'!")


print("\n" + "="*40)
print("4. MONTH-WISE EVALUATION (SCALED METRICS)")
print("="*40)

print("Making predictions...")
predictions_scaled = model.predict([X_seq_test, X_geo_test]).flatten()

# Create a DataFrame with the predictions
test_df = pd.DataFrame({
    'date': pd.to_datetime(target_dates_test),
    'actual_scaled': y_test,
    'predicted_scaled': predictions_scaled
})

# Add a month column to make filtering easy
test_df['month'] = test_df['date'].dt.month

months_to_evaluate = {
    6: "June",
    7: "July",
    8: "August",
    9: "September"
}

print(f"\nOVERALL MONTH-WISE EVALUATION (All Rajasthan Regions Combined)")
print(f"{'-'*60}")
print(f"{'Month':<15} | {'Data Points':<12} | {'MAE (Scaled)':<12} | {'RMSE (Scaled)':<12}")
print(f"{'-'*60}")

for month_num, month_name in months_to_evaluate.items():
    # Filter the dataframe to ONLY include rows for this specific month
    month_data = test_df[test_df['month'] == month_num]
    
    if not month_data.empty:
        mae = mean_absolute_error(month_data['actual_scaled'], month_data['predicted_scaled'])
        rmse = np.sqrt(mean_squared_error(month_data['actual_scaled'], month_data['predicted_scaled']))
        count = len(month_data)
        
        print(f"{month_name:<15} | {count:<12} | {mae:<12.4f} | {rmse:<12.4f}")

print(f"{'-'*60}")