import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load time-series data
# Use absolute path for CSV
csv_path = r'c:\Users\Nirmal Shinde\Desktop\vaishnavi\sales_data.csv'
df = pd.read_csv(csv_path, parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Resample weekly & calculate total sales
weekly_sales = df['Sales'].resample('W').sum()

# Line Plot: Weekly trend
plt.figure(figsize=(10, 5))
sns.lineplot(x=weekly_sales.index, y=weekly_sales.values, marker='o')
plt.title('Weekly Sales Trend')
plt.xlabel('Week')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Add rolling average (4 weeks)
plt.figure(figsize=(10, 5))
weekly_sales.plot(label='Actual', alpha=0.5)
weekly_sales.rolling(4).mean().plot(label='4-Week Avg', linewidth=2)
plt.legend()
plt.title('Sales with Rolling Average')
plt.xlabel('Week')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

# Anomaly detection using z-score
z_scores = zscore(weekly_sales.values)
anomaly_threshold = 2 # change as needed
anomalies = weekly_sales[abs(z_scores) > anomaly_threshold]
