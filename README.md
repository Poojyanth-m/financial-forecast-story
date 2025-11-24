# 📈 Financial Forecast Story — Time Series Forecasting with SARIMAX

## ✅ Project Overview
This mini-project builds an **end-to-end financial forecasting pipeline** using **SARIMAX**, producing:
- Monthly synthetic financial data  
- Cleaned and prepared time-series  
- Train/validation/test split (70% / 15% / 15%)  
- SARIMAX time series model with automatic order selection  
- Forecasts with confidence intervals  
- Evaluation metrics (MAE, RMSE, MAPE)  
- Automatically generated narrative explaining the forecast  

This project is simple, lightweight, and fully runnable within minutes — ideal for academic mini-projects.

## 📂 **Project Structure**
financial-forecast-story/
│
├── data/                         # synthetic_monthly.csv stored here
├── src/
│   ├── generate_synthetic_data.py
│   ├── data_prep.py
│   ├── train_test_split.py
│   ├── model_sarimax.py
│   ├── evaluate.py
│   ├── narrative.py
│   └── run_pipeline.py           # MAIN SCRIPT
│
├── results/                      # automatically generated outputs
│   ├── forecasts.csv
│   ├── metrics.json
│   ├── model_summary.txt
│   ├── narrative.txt
│   └── forecast_plot.png

├── scripts/
│   └── plot_forecast.py          # optional visualization script

├── requirements.txt
└── README.md

## 🚀 **How to Run the Project**

### **1. Create & activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate

```
### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the full forecasting pipeline**

```bash
python src/run_pipeline.py
```

This generates all results in the `results/` folder.

## 📊 **Generated Outputs**

After running the pipeline, you will find:

| File                        | Description                              |
| --------------------------- | ---------------------------------------- |
| **forecasts.csv**           | Forecasted values + confidence intervals |
| **metrics.json**            | MAE, RMSE, MAPE for model performance    |
| **model_summary.txt**       | SARIMAX internal diagnostic summary      |
| **narrative.txt**           | Automatically generated text explanation |
| **prepared_series.csv**     | Cleaned and processed time-series        |
| **train / val / test CSVs** | Split datasets                           |

Example metrics (your values may differ):

```json
{
  "MAE": 13.55,
  "RMSE": 17.50,
  "MAPE": 9.25
}
```

## ✍ **Narrative Example**

From `results/narrative.txt`:

```
Executive summary:
- Detected trend: upward
- Model: order=(0, 1, 3), seasonal=(0, 1, 1, 12)
- MAE: 13.56
- RMSE: 17.50
- MAPE: 9.25%
Recommendations: Monitor forecast monthly and update model if performance degrades.
```

## 🧹 **Optional: Plot the Forecast**

Generate a PNG plot using:

```bash
python scripts/plot_forecast.py
```

Output saved to:

```
results/forecast_plot.png
```

## 📘 **Methodology Summary**

1. **Data Generation**
   Synthetic monthly data from 2015–2024 with trend + seasonality + noise.

2. **Data Preparation**

   * Parse dates
   * Sort and clean
   * Set monthly frequency
   * Interpolate missing values

3. **Train/Val/Test Split**
   Ratio: **70% / 15% / 15%**.

4. **SARIMAX Model**

   * Small grid search for p, q, P, Q
   * Seasonal period m = 12
   * Best model selected via AIC

5. **Forecasting**

   * Forecast horizon = max(12, test length)
   * Confidence intervals included

6. **Evaluation Metrics**

   * Mean Absolute Error (MAE)
   * Root Mean Squared Error (RMSE)
   * Mean Absolute Percentage Error (MAPE)

7. **Narrative Generation**
   Natural-language summary of model + trends + recommendations.

## 🔮 **Future Enhancements**

* Prophet & Auto-ARIMA model comparison
* Streamlit dashboard UI
* Real financial dataset support
* Rolling window cross-validation
