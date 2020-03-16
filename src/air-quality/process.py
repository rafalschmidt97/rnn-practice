import configparser
from functools import reduce
from operator import add

import matplotlib.pyplot as plt
import pandas as pd

# Loading configuration
config = configparser.ConfigParser()
config.read('config.ini')

processing_percentage = config.getint('APP', 'PROCESSING_PERCENTAGE')

# Selecting data
df = pd.read_csv('data/simplified_small.csv', sep=',', usecols=[
    'Date Local', 'Year', 'Month', 'Day', 'NO2 Max AQI', 'O3 Max AQI', 'SO2 Max AQI', 'CO Max AQI'
])

# Merge fields from multiple columns
df = df.groupby('Date Local').agg(list)
df.reset_index(drop=False, inplace=True)
df['AvgNO2QI'] = df['NO2 Max AQI'].apply(lambda x: reduce(add, x) / len(x))
df['AvgO3AQI'] = df['O3 Max AQI'].apply(lambda x: reduce(add, x) / len(x))
df['AvgSO2AQI'] = df['SO2 Max AQI'].apply(lambda x: reduce(add, x) / len(x))
df['AvgCOAQI'] = df['CO Max AQI'].apply(lambda x: reduce(add, x) / len(x))
df['Year'] = df['Year'].apply(lambda x: x[0])
df['Month'] = df['Month'].apply(lambda x: x[0])
df['Day'] = df['Day'].apply(lambda x: x[0])
df.drop(['NO2 Max AQI', 'O3 Max AQI', 'SO2 Max AQI', 'CO Max AQI'], 1, inplace=True)

# Plot final chart
plt.figure(figsize=(10, 6))
df['AvgNO2QI'].plot(color='r', linewidth=1.5, label='NO2')
df['AvgO3AQI'].plot(color='b', linewidth=1.5, label='O3')
df['AvgSO2AQI'].plot(color='g', linewidth=1.5, label='SO2')
df['AvgSO2AQI'].plot(color='c', linewidth=1.5, label='CO')
plt.legend()
plt.show()

# # Save data
# df.to_csv(f'data/processed.csv', index=False)
df.to_csv(f'data/processed_small.csv', index=False)
processing_size = int(len(df) * (0.01 * processing_percentage))
# df[-processing_size:].to_csv(f'data/processed_{processing_percentage}.csv', index=False)
df[-processing_size:].to_csv(f'data/processed_small_{processing_percentage}.csv', index=False)
