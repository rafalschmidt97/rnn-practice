import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/conventional_weather_stations_inmet_brazil_1961_2019.csv', sep=';', dtype=str, usecols=[
    'Estacao', 'Data', 'Hora', 'Precipitacao', 'TempBulboSeco', 'TempBulboUmido', 'TempMaxima', 'TempMinima',
    'UmidadeRelativa', 'PressaoAtmEstacao', 'PressaoAtmMar', 'DirecaoVento', 'VelocidadeVento', 'Insolacao',
    'Nebulosidade', 'Evaporacao Piche', 'Temp Comp Media', 'Umidade Relativa Media', 'Velocidade do Vento Media'
])

df.rename(columns={'Estacao': 'Code', 'Data': 'Date', 'Hora': 'Hour',
                   'Precipitacao': 'Precipitation', 'TempBulboSeco': 'DryBulbTemp',
                   'TempBulboUmido': 'WetBulbTemp', 'TempMaxima': 'MaxTemp',
                   'TempMinima': 'MinTemp', 'UmidadeRelativa': 'Humidity',
                   'PressaoAtmEstacao': 'PressureStation', 'PressaoAtmMar': 'PressureSea',
                   'DirecaoVento': 'WindDirection', 'VelocidadeVento': 'WindSpeed',
                   'Insolacao': 'Insolation', 'Nebulosidade': 'Cloudiness',
                   'Evaporacao Piche': 'Evaporation', 'Temp Comp Media': 'AvgCompTemp',
                   'Umidade Relativa Media': 'AvgRelHumidity',
                   'Velocidade do Vento Media': 'AvgWindSpeed'}, inplace=True)

# Connected the date
df['Date'] = df['Date'] + ' ' + df['Hour'].apply(lambda x: x[0:2] + ':' + x[2:4] + ':00')
df.drop('Hour', 1, inplace=True)

# Show a chart with the most records
plt.style.use('fivethirtyeight')
_, axis = plt.subplots(figsize=(12, 9))
df['Code'].value_counts().head(5).plot.bar(color='green')  # most = 82331
axis.set_xticklabels(axis.get_xticklabels(), rotation=25)
plt.title('The most active/precious stations')
plt.show()
print(df['Code'].value_counts().head(5))
# df.drop('Code', 1, inplace=True)

# Save selected data
df.to_csv('data/simplified.csv', sep=';', index=False)
df[df['Code'] == '82331'].to_csv('data/simplified_small.csv', sep=';', index=False)
