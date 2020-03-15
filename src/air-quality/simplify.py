import matplotlib.pyplot as plt
import pandas as pd
from aqi.constants import POLLUTANT_CO_1H, POLLUTANT_NO2_1H, POLLUTANT_SO2_1H, POLLUTANT_O3_1H, ALGO_EPA, \
    POLLUTANT_CO_8H, POLLUTANT_O3_8H
from aqi import to_iaqi

df = pd.read_csv('data/pollution_us_2000_2016.csv', sep=',', usecols=[
    'State Code', 'County Code', 'Site Num', 'Address', 'State', 'County', 'City', 'Date Local',
    'NO2 Units', 'NO2 Mean', 'NO2 1st Max Value', 'NO2 1st Max Hour', 'NO2 AQI',
    'O3 Units', 'O3 Mean', 'O3 1st Max Value', 'O3 1st Max Hour', 'O3 AQI',
    'SO2 Units', 'SO2 Mean', 'SO2 1st Max Value', 'SO2 1st Max Hour', 'SO2 AQI',
    'CO Units', 'CO Mean', 'CO 1st Max Value', 'CO 1st Max Hour', 'CO AQI'
])
print(f"Length: {len(df)}")

# Show a chart with the most records
plt.style.use('fivethirtyeight')
_, axis = plt.subplots(figsize=(12, 9))
df['City'].value_counts().head(5).plot.bar(color='red')
axis.set_xticklabels(axis.get_xticklabels(), rotation=25)
plt.title('City')
plt.show()
# print(df['City'].value_counts().head(5))  # New York and Los Angeles are two biggest players here

# Show a chart with two cities addresses
city_df = df[(df['City'] == 'New York') | (df['City'] == 'Los Angeles')]
plt.style.use('fivethirtyeight')
_, axis = plt.subplots(figsize=(12, 9))
city_df['Address'].value_counts().head(5).plot.bar(color='red')
axis.set_xticklabels(axis.get_xticklabels(), rotation=25)
plt.title('Address')
plt.show()
# print(city_df['Address'].value_counts().head(5))
# 1630 N MAIN ST, LOS ANGELES (25225) is a good pick within a city with many records (42241)
# so I can focus on the location prediction later as well.

# Show a chart with address missing data
simplified_df = city_df[city_df['Address'] == '1630 N MAIN ST, LOS ANGELES']
plt.style.use('fivethirtyeight')
_, axis = plt.subplots(figsize=(12, 9))
simplified_df.isna().sum().plot.bar(color='red')
plt.ylim(0, len(simplified_df))
axis.set_xticklabels(axis.get_xticklabels(), rotation=25)
plt.title('Missing value')
plt.show()
# print(simplified_df.isna().sum())  # SO2 AQI and CO AQI contain bunch of nulls (almost half)

# print(to_iaqi(POLLUTANT_SO2_1H, '3.0', algo=ALGO_EPA))  # 4
# print(to_iaqi(POLLUTANT_CO_8H, '1.6', algo=ALGO_EPA))  # 18
# print(to_iaqi(POLLUTANT_NO2_1H, '48.0', algo=ALGO_EPA))  # 45
# print(to_iaqi(POLLUTANT_O3_8H, '0.027', algo=ALGO_EPA))  # 23

# Recalculate AQI
pd.options.mode.chained_assignment = None  # disable false warning for copying
simplified_df['NO2 Max AQI'] = simplified_df['NO2 1st Max Value'] \
    .apply(lambda x: to_iaqi(POLLUTANT_NO2_1H, str(x), algo=ALGO_EPA))
simplified_df['O3 Max AQI'] = simplified_df['O3 1st Max Value'] \
    .apply(lambda x: to_iaqi(POLLUTANT_O3_8H, str(x), algo=ALGO_EPA))
simplified_df['SO2 Max AQI'] = simplified_df['SO2 1st Max Value'] \
    .apply(lambda x: to_iaqi(POLLUTANT_SO2_1H, str(x), algo=ALGO_EPA))
simplified_df['CO Max AQI'] = simplified_df['CO 1st Max Value'] \
    .apply(lambda x: to_iaqi(POLLUTANT_CO_8H, str(x), algo=ALGO_EPA))

simplified_df[['NO2 Max AQI', 'O3 Max AQI', 'SO2 Max AQI', 'CO Max AQI']] = \
    simplified_df[['NO2 Max AQI', 'O3 Max AQI', 'SO2 Max AQI', 'CO Max AQI']].apply(pd.to_numeric, axis=1)

# Select the data
final_df = simplified_df[['Date Local', 'NO2 Max AQI', 'O3 Max AQI', 'SO2 Max AQI', 'CO Max AQI']]
final_df.reset_index(drop=True, inplace=True)

final_df['Date Local'] = pd.to_datetime(final_df['Date Local'], infer_datetime_format=True)
final_df['Year'] = final_df['Date Local'].apply(lambda x: x.year)
final_df['Month'] = final_df['Date Local'].apply(lambda x: x.month)
final_df['Day'] = final_df['Date Local'].apply(lambda x: x.day)
# print(final_df.head())

# Plot one year chart

plot_df = final_df[final_df['Year'] == 2015].set_index('Date Local')
plt.figure(figsize=(10, 6))
plot_df['NO2 Max AQI'].plot(color='r', linewidth=1.5, label='NO2')
plot_df['O3 Max AQI'].plot(color='b', linewidth=1.5, label='O3')
plot_df['SO2 Max AQI'].plot(color='g', linewidth=1.5, label='SO2')
plot_df['CO Max AQI'].plot(color='c', linewidth=1.5, label='CO')
plt.legend()
plt.show()

final_df.to_csv('data/simplified_small.csv', sep=',')
