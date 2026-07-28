# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import xarray as xr
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM, Dropout, BatchNormalization, Concatenate, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
from dask.diagnostics import ProgressBar

# Verify GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Using GPU: {physical_devices[0]}")

print("\n" + "="*40)
print("1. DATA LOADING & VECTORIZED PREPROCESSING")
print("="*40)

file_path = 'data/India_Meteo_Combined_Final.nc'

print("Loading NetCDF dataset...")
ds = xr.open_dataset(file_path).chunk({'time': 500})

print("Filtering spatial coordinates for Rajasthan...")
mask_lat = (ds.lat >= 23.08) & (ds.lat <= 30.24)
mask_lon = (ds.lon >= 69.45) & (ds.lon <= 78.32)
rajasthan_ds = ds.where(mask_lat & mask_lon, drop=True)

print("Performing fast vectorized Feature Engineering & Resampling...")
ds_monthly_rain = rajasthan_ds['rain'].resample(time='ME').sum()
ds_monthly_temp = rajasthan_ds[['tmax', 'tmin']].resample(time='ME').mean()

with ProgressBar():
    print("Crunching spatial data into memory:")
    ds_monthly = xr.merge([ds_monthly_rain, ds_monthly_temp]).compute()

df = ds_monthly.to_dataframe().dropna().reset_index()
df.sort_values(by=['lat', 'lon', 'time'], inplace=True)

print("\nInjecting new features (Lags & Seasonality)...")
df['tmean'] = (df['tmax'] + df['tmin']) / 2.0
df['t_range'] = df['tmax'] - df['tmin']
df['rain_lag_1'] = df.groupby(['lat', 'lon'])['rain'].shift(1)

df['month_sin'] = np.sin(2 * np.pi * df['time'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['time'].dt.month / 12)

df.dropna(inplace=True)

features = ['rain', 'tmax', 'tmin', 'tmean', 't_range', 'rain_lag_1', 'month_sin', 'month_cos']
num_features = len(features)

print("Applying RobustScaler...")
scaler = RobustScaler()
df[features] = scaler.fit_transform(df[features])
df.set_index('time', inplace=True)

print("\n" + "="*40)
print("2. SEQUENCE GENERATION & CHRONOLOGICAL SPLIT")
print("="*40)
SEQUENCE_LENGTH = 108 

def create_sequences_and_split(df, seq_length, feature_cols):
    X_train, X_geo_train, y_train, dates_train = [], [], [], []
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
        
        X_train.extend(loc_X[:split_idx])
        X_geo_train.extend([[lat, lon]] * split_idx)
        y_train.extend(loc_y[:split_idx])
        dates_train.extend(loc_dates[:split_idx])
        
        X_test.extend(loc_X[split_idx:])
        X_geo_test.extend([[lat, lon]] * (len(loc_X) - split_idx))
        y_test.extend(loc_y[split_idx:])
        dates_test.extend(loc_dates[split_idx:])
            
    return (np.array(X_train), np.array(X_geo_train), np.array(y_train), np.array(dates_train),
            np.array(X_test), np.array(X_geo_test), np.array(y_test), np.array(dates_test))

X_seq_train, X_geo_train, y_train, target_dates_train, X_seq_test, X_geo_test, y_test, target_dates_test = create_sequences_and_split(df, SEQUENCE_LENGTH, features)


print("\n" + "="*40)
print("3. CNN-LSTM MODEL ARCHITECTURE & TRAINING")
print("="*40)

input_seq = Input(shape=(SEQUENCE_LENGTH, num_features), name="time_series_input")
input_geo = Input(shape=(2,), name="geo_input")

x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_seq)
x = MaxPooling1D(pool_size=2)(x)

x = LSTM(128, return_sequences=True)(x)
x = Dropout(0.3)(x)
x = LSTM(64)(x)
x = Dropout(0.2)(x)

merged = Concatenate()([x, input_geo])

x = Dense(32, activation='relu')(merged)
x = BatchNormalization()(x)
output = Dense(1, activation='linear', name="rainfall_prediction")(x)

model = Model(inputs=[input_seq, input_geo], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='huber', metrics=['mae'])

print("Starting Training with Strict Early Stopping & Plateau Reduction...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
]

history = model.fit(
    [X_seq_train, X_geo_train], y_train,
    epochs=100,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,  
    verbose=1
)

# -------------------------------------------------------------------
# SAVE THE TRAINED MODEL TO HARD DRIVE
# -------------------------------------------------------------------
print("\nSaving the optimized model to hard drive...")
model.save("optimized_monsoon_model.keras")
print("Model saved successfully as 'optimized_monsoon_model.keras'!")
# -------------------------------------------------------------------

print("\n" + "="*40)
print("4. QUICK EVALUATION (TABLE 3 SCALED METRICS)")
print("="*40)

print("Making predictions on test data...")
predictions_scaled = model.predict([X_seq_test, X_geo_test]).flatten()

test_df = pd.DataFrame({
    'date': pd.to_datetime(target_dates_test),
    'lat': X_geo_test[:, 0],
    'lon': X_geo_test[:, 1],
    'actual_scaled': y_test,
    'predicted_scaled': predictions_scaled
})

table3_locations = [
    {"zone": "North-West Desert", "display_coord": "29 deg 00'N 73 deg 00'E", "target_lat": 29.00, "target_lon": 73.00},
    {"zone": "Central Aravalli Hill Region", "display_coord": "26 deg 00'N 74 deg 50'E", "target_lat": 26.00, "target_lon": 74.833}, 
    {"zone": "Eastern Plain", "display_coord": "26 deg 50'N 75 deg 00'E", "target_lat": 26.833, "target_lon": 75.00}, 
    {"zone": "South-Eastern Plateau Region", "display_coord": "25 deg 25'N 75 deg 50'E", "target_lat": 25.416, "target_lon": 75.833} 
]

print(f"\nTABLE 3: Prediction Errors in SCALED METRICS per month")
print(f"{'-'*110}")
print(f"{'Zone Name':<30} | {'Coordinates':<18} | {'June':<12} | {'July':<12} | {'August':<12} | {'September':<12}")
print(f"{'':<30} | {'':<18} | {'MAE':<5} {'RMSE':<6} | {'MAE':<5} {'RMSE':<6} | {'MAE':<5} {'RMSE':<6} | {'MAE':<5} {'RMSE':<6}")
print(f"{'-'*110}")

for loc in table3_locations:
    available_lats = test_df['lat'].unique()
    available_lons = test_df['lon'].unique()
    nearest_lat = available_lats[np.argmin(np.abs(available_lats - loc["target_lat"]))]
    nearest_lon = available_lons[np.argmin(np.abs(available_lons - loc["target_lon"]))]
    
    loc_df = test_df[(test_df['lat'] == nearest_lat) & (test_df['lon'] == nearest_lon)]
    
    metrics = {}
    for month in [6, 7, 8, 9]:
        month_df = loc_df[loc_df['date'].dt.month == month]
        if not month_df.empty:
            rmse = np.sqrt(mean_squared_error(month_df['actual_scaled'], month_df['predicted_scaled']))
            mae = mean_absolute_error(month_df['actual_scaled'], month_df['predicted_scaled'])
            metrics[month] = {"rmse": rmse, "mae": mae}
        else:
            metrics[month] = {"rmse": 0.0, "mae": 0.0}

    print(f"{loc['zone']:<30} | {loc['display_coord']:<18} | "
          f"{metrics[6]['mae']:<5.4f} {metrics[6]['rmse']:<6.4f} | "
          f"{metrics[7]['mae']:<5.4f} {metrics[7]['rmse']:<6.4f} | "
          f"{metrics[8]['mae']:<5.4f} {metrics[8]['rmse']:<6.4f} | "
          f"{metrics[9]['mae']:<5.4f} {metrics[9]['rmse']:<6.4f}")
print(f"{'-'*110}")
print("\nProcess Complete! You can now use 'optimized_monsoon_model.keras' in your other scripts.")