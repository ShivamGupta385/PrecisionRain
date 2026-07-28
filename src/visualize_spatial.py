# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
from dask.diagnostics import ProgressBar

# NEW IMPORTS FOR MAPPING
import geopandas as gpd
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Verify GPU and prevent memory allocation crashes
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
print("3. LOADING PRE-TRAINED MODEL & PREDICTING")
print("="*40)

model = load_model("optimized_monsoon_model.keras")
# Added batch_size to prevent memory crashes
predictions_scaled = model.predict([X_seq_test, X_geo_test], batch_size=2048).flatten()

test_df = pd.DataFrame({
    'date': pd.to_datetime(target_dates_test),
    'lat': X_geo_test[:, 0],
    'lon': X_geo_test[:, 1],
    'actual': y_test,
    'predicted': predictions_scaled
})

test_df['absolute_error'] = np.abs(test_df['actual'] - test_df['predicted'])
test_df['squared_error'] = (test_df['actual'] - test_df['predicted'])**2


print("\n" + "="*40)
print("4. GENERATING MASKED SPATIAL HEATMAPS (FIGURE 7)")
print("="*40)

print("Loading Rajasthan Shapefile...")
map_data = gpd.read_file('data/shapefiles/Indian_States.shp')

# --- SMART SEARCH: Find the correct column name automatically ---
target_state = 'Rajasthan'
state_column = None

for col in map_data.columns:
    if map_data[col].dtype == 'object' and target_state in map_data[col].values:
        state_column = col
        break

if state_column is None:
    print(f"ERROR: Could not find '{target_state}' in any column. The available columns are: {map_data.columns}")
    exit()

print(f"Successfully located Rajasthan in column: '{state_column}'")
# ----------------------------------------------------------------

rajasthan_map = map_data[map_data[state_column] == target_state] 
rajasthan_geom = rajasthan_map.geometry.iloc[0]

# Setup for the 8 maps
months = [6, 7, 8, 9]
month_names = ['June', 'July', 'August', 'September']
letters = [['(A)', '(B)'], ['(C)', '(D)'], ['(E)', '(F)'], ['(G)', '(H)']]

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 24))
fig.tight_layout(pad=6.0)

for row_idx, month in enumerate(months):
    month_df = test_df[test_df['date'].dt.month == month]
    
    spatial_data = month_df.groupby(['lat', 'lon']).agg(
        mae=('absolute_error', 'mean'),
        mse=('squared_error', 'mean')
    ).reset_index()
    
    spatial_data['rmse'] = np.sqrt(spatial_data['mse'])
    
    lons = spatial_data['lon'].values
    lats = spatial_data['lat'].values
    grid_lon, grid_lat = np.mgrid[lons.min():lons.max():500j, lats.min():lats.max():500j]
    
    # --- HELPER LOGIC FOR THE COOKIE CUTTER ---
    if hasattr(rajasthan_geom, 'geoms'):
        paths = [Path(np.asarray(poly.exterior.coords)) for poly in rajasthan_geom.geoms]
    else:
        paths = [Path(np.asarray(rajasthan_geom.exterior.coords))]
    compound_path = Path.make_compound_path(*paths)
    
    # ==========================================
    # LEFT COLUMN: MAE PLOT
    # ==========================================
    ax_mae = axes[row_idx, 0]
    mae_errors = spatial_data['mae'].values
    grid_mae = griddata((lons, lats), mae_errors, (grid_lon, grid_lat), method='cubic')
    
    contour_mae = ax_mae.contourf(grid_lon, grid_lat, grid_mae, levels=50, cmap='turbo')
    
    # Apply the mask (VERSION-PROOF FIX)
    patch_mae = PathPatch(compound_path, transform=ax_mae.transData, facecolor='none')
    ax_mae.add_patch(patch_mae)
    
    if hasattr(contour_mae, 'collections'):
        for collection in contour_mae.collections:
            collection.set_clip_path(patch_mae)
    else:
        contour_mae.set_clip_path(patch_mae) # For Matplotlib 3.8+
        
    rajasthan_map.plot(ax=ax_mae, facecolor='none', edgecolor='black', linewidth=1.5)
    
    cbar_mae = fig.colorbar(contour_mae, ax=ax_mae, fraction=0.046, pad=0.04)
    cbar_mae.set_label('MAE (Scaled)', size=10)
    ax_mae.scatter(lons, lats, color='black', s=1, alpha=0.2) 
    ax_mae.set_title(f"{letters[row_idx][0]} {month_names[row_idx]} - MAE", fontsize=14, y=-0.15)
    ax_mae.grid(True, linestyle='--', alpha=0.5)
    
    # ==========================================
    # RIGHT COLUMN: RMSE PLOT
    # ==========================================
    ax_rmse = axes[row_idx, 1]
    rmse_errors = spatial_data['rmse'].values
    grid_rmse = griddata((lons, lats), rmse_errors, (grid_lon, grid_lat), method='cubic')
    
    contour_rmse = ax_rmse.contourf(grid_lon, grid_lat, grid_rmse, levels=50, cmap='turbo')
    
    # Apply the mask (VERSION-PROOF FIX)
    patch_rmse = PathPatch(compound_path, transform=ax_rmse.transData, facecolor='none')
    ax_rmse.add_patch(patch_rmse)
    
    if hasattr(contour_rmse, 'collections'):
        for collection in contour_rmse.collections:
            collection.set_clip_path(patch_rmse)
    else:
        contour_rmse.set_clip_path(patch_rmse) # For Matplotlib 3.8+
        
    rajasthan_map.plot(ax=ax_rmse, facecolor='none', edgecolor='black', linewidth=1.5)
    
    cbar_rmse = fig.colorbar(contour_rmse, ax=ax_rmse, fraction=0.046, pad=0.04)
    cbar_rmse.set_label('RMSE (Scaled)', size=10)
    ax_rmse.scatter(lons, lats, color='black', s=1, alpha=0.2)
    ax_rmse.set_title(f"{letters[row_idx][1]} {month_names[row_idx]} - RMSE", fontsize=14, y=-0.15)
    ax_rmse.grid(True, linestyle='--', alpha=0.5)

output_filename = "images/spatial_heatmaps.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nSUCCESS: Masked 8-panel heatmap saved as '{output_filename}' in your directory!")