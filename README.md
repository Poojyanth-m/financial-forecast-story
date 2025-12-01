# Financial Forecast Story

Time series forecasting application using SARIMAX models.

## Features

- Web interface for time series forecasting
- CSV upload with automatic column detection
- SARIMAX model optimization
- Forecast visualization with confidence intervals
- Performance metrics (MAE, RMSE, MAPE)
- Automated result summaries

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Web Application
```bash
python3 -m streamlit run app.py
```

### Command Line
```bash
python src/run_pipeline.py
```

## Data Format

CSV with date and numeric value columns. Minimum 50 observations recommended.

## Dependencies

- numpy
- pandas
- matplotlib
- statsmodels
- scikit-learn
- streamlit
- jinja2

