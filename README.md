# Indian Monsoon Rainfall Forecasting (1951–2024)
### Multivariate CNN-LSTM Hybrid Architecture on IMD Gridded Data

This project implements a high-precision deep learning pipeline to forecast daily rainfall in Kerala, India. By fusing 74 years of historical temperature and precipitation data, the model captures the complex thermal gradients that drive the Indian Summer Monsoon (ISM).

---

## Key Highlights (Proof of Work)
* **Data Scale:** Processed **9GB** of raw IMD binary data spanning **27,000+ days**.
* **Architecture:** Hybrid **Conv1D + LSTM** network.
* **Accuracy:** Achieved a Validation **MAE of 2.67 mm** and **RMSE of 4.28 mm**.
* **Validation:** Model performance was benchmarked against **Official 2024 IMD Monsoon Onset Dates**, successfully capturing pre-monsoon surges.

---

## Technical Implementation

### 1. Data Engineering & Spatial Alignment
The primary challenge involved aligning multi-resolution datasets:
* **Rainfall:** 0.25° x 0.25° resolution.
* **Temperature (Tmax/Tmin):** 1.0° x 1.0° resolution.
I implemented **Bilinear Interpolation** using `xarray` to upsample temperature data to a unified 0.25° grid, creating a multi-source temporal cube.

### 2. Feature Engineering
Beyond raw rainfall, the model utilizes:
* **Diurnal Temperature Range (DTR):** Calculated as `Tmax - Tmin`.
* **Cyclical Encoding:** Sine/Cosine transformation of the 'Day of Year' to capture the 365-day monsoon periodicity.
* **Lag Features:** 1-day shifted rainfall to capture short-term persistence.

### 3. Optimization
* **Loss Function:** Used **Huber Loss** to maintain robustness against extreme precipitation outliers.
* **Regularization:** Implemented **Early Stopping** (patience=12) and **Learning Rate Decay** to prevent overfitting.

---

## 📊 Results & Diagnostics
The repository includes comprehensive diagnostic plots:
* **2024 Forecast vs. IMD Onset:** Visual proof of the model capturing the monsoon arrival.
* **Monthly Error Breakdown:** Analysis of how the model performs across dry and wet seasons.
* **Residual Analysis:** Histogram and scatter plots proving a zero-biased error distribution.
* **Feature Importance:** Permutation shuffling revealed **Tmin** as a top-3 predictor for regional rainfall.

---

## 📂 Repository Structure
```text
├── Monsoon_Prediction_Full_Pipeline.ipynb  # Main Notebook with all outputs
├── requirements.txt                        # Environment dependencies
├── .gitignore                              # Prevents uploading 9GB .nc files
└── README.md                               # Project documentation
