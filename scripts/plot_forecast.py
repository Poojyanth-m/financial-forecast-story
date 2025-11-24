import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

r = Path('results')
hist = pd.read_csv(r / 'prepared_series.csv', parse_dates=['date']).set_index('date')
fc = pd.read_csv(r / 'forecasts.csv', parse_dates=['date']).set_index('date')

plt.figure(figsize=(10,5))
plt.plot(hist.index, hist['value'], label='History')
plt.plot(fc.index, fc['forecast'], label='Forecast')
plt.fill_between(fc.index, fc['ci_lower'], fc['ci_upper'], alpha=0.2)
plt.legend()
plt.title('History and Forecast')
plt.tight_layout()
plt.savefig(r / 'forecast_plot.png')
print('Saved plot to results/forecast_plot.png')
