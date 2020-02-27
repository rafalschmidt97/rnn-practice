import pandas as pd

df = pd.read_csv('data/conventional_weather_stations_inmet_brazil_1961_2019.csv', sep=';', dtype=str)
df.dropna(how='all', axis='columns', inplace=True)
df.tail(1000).to_csv('data/conventional_weather_stations_inmet_brazil_1961_2019_tail.csv', sep=";", index=False)
