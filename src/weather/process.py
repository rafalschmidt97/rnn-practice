import configparser
import time

import pandas as pd

PROGRESS = 0


def step():
    global PROGRESS
    PROGRESS += 1
    print(f'Progress: {PROGRESS}')


def resolve_location(code):
    _, latitude, longitude, altitude = s_df[s_df['Code'] == code].iloc[0]
    return pd.Series([latitude, longitude, altitude])


# Loading configuration
config = configparser.ConfigParser()
config.read('config.ini')

processing_percentage = config.getint('APP', 'PROCESSING_PERCENTAGE')

step()

# Selecting data
w_df = pd.read_csv('data/conventional_weather_stations_inmet_brazil_1961_2019_tail.csv', sep=';', dtype=str)
w_df.rename(columns={'Estacao': 'Code', 'Data': 'Date', 'Hora': 'Hour',
                     'Precipitacao': 'Precipitation', 'TempBulboSeco': 'DryBulbTemp',
                     'TempBulboUmido': 'WetBulbTemp', 'TempMaxima': 'MaxTemp',
                     'TempMinima': 'MinTemp', 'UmidadeRelativa': 'Humidity',
                     'PressaoAtmEstacao': 'PressureStation', 'PressaoAtmMar': 'PressureSea',
                     'DirecaoVento': 'WindDirection', 'VelocidadeVento': 'WindSpeed',
                     'Insolacao': 'Insolation', 'Nebulosidade': 'Cloudiness',
                     'Evaporacao Piche': 'Evaporation', 'Temp Comp Media': 'AvgCompTemp',
                     'Umidade Relativa Media': 'AvgRelHumidity',
                     'Velocidade do Vento Media': 'AvgWindSpeed'}, inplace=True)

step()

# Sort data by date (connect two fields)
w_df['Date'] = w_df['Date'] + ' ' + w_df['Hour'].apply(lambda x: x[0:2] + ':' + x[2:4] + ':00')
w_df.drop('Hour', 1, inplace=True)
w_df['Date'] = pd.to_datetime(w_df['Date'], infer_datetime_format=True)
w_df.sort_values(by=['Date'], ascending=True, inplace=True)
w_df['Day'] = w_df['Date'].apply(lambda x: x.day)
w_df['Month'] = w_df['Date'].apply(lambda x: x.month)
w_df['Year'] = w_df['Date'].apply(lambda x: x.year)
w_df['Hour'] = w_df['Date'].apply(lambda x: x.hour)
w_df['Minutes'] = w_df['Date'].apply(lambda x: x.minute)
w_df['Date'] = w_df['Date'].apply(lambda x: str(int(time.mktime(x.timetuple()) * 1000)))

step()

# Resolve location
s_df = pd.read_csv('data/weather_stations_codes.csv', sep=';', dtype=str,
                   usecols=['Código', 'Latitude', 'Longitude', 'Altitude'])
s_df.rename(columns={'Código': 'Code'}, inplace=True)

w_df[['Latitude', 'Longitude', 'Altitude']] = w_df['Code'].apply(resolve_location)
w_df.drop('Code', 1, inplace=True)

step()

# Merge fields
w_df['Humidity0'], w_df['Humidity12'], w_df['Humidity18'] = [0, 0, 0]
w_df['Pressure0'], w_df['Pressure12'], w_df['Pressure18'] = [0.0, 0.0, 0.0]
w_df['WindDirection0'], w_df['WindDirection12'], w_df['WindDirection18'] = [0, 0, 0]
w_df['WindSpeed0'], w_df['WindSpeed12'], w_df['WindSpeed18'] = [0.0, 0.0, 0.0]
w_df['Cloudiness0'], w_df['Cloudiness12'], w_df['Cloudiness18'] = [0.0, 0.0, 0.0]

for index, morning in w_df[1:11].iterrows():
    if morning['Hour'] != 0:
        continue

    midday = w_df.iloc[index + 1]
    evening = w_df.iloc[index + 2]

    # Migrate values to one row
    w_df.at[index, 'Precipitation'] = midday['Precipitation']  # todo: there is evaporation as well (but many nulls)
    w_df.at[index, 'MinTemp'] = midday['MinTemp']
    w_df.at[index, 'Humidity0'], w_df.at[index, 'Humidity12'], w_df.at[index, 'Humidity18'] = \
        [morning['Humidity'], midday['Humidity'], evening['Humidity']]
    w_df.at[index, 'Pressure0'], w_df.at[index, 'Pressure12'], w_df.at[index, 'Pressure18'] = \
        [morning['PressureStation'], midday['PressureStation'], evening['PressureStation']]
    w_df.at[index, 'WindDirection0'], w_df.at[index, 'WindDirection12'], w_df.at[index, 'WindDirection18'] = \
        [morning['WindDirection'], midday['WindDirection'], evening['WindDirection']]
    w_df.at[index, 'WindSpeed0'], w_df.at[index, 'WindSpeed12'], w_df.at[index, 'WindSpeed18'] = \
        [morning['WindSpeed'], midday['WindSpeed'], evening['WindSpeed']]
    w_df.at[index, 'Cloudiness0'], w_df.at[index, 'Cloudiness12'], w_df.at[index, 'Cloudiness18'] = \
        [morning['Cloudiness'], midday['Cloudiness'], evening['Cloudiness']]

w_df.drop(['DryBulbTemp', 'WetBulbTemp', 'Evaporation', 'AvgCompTemp', 'AvgRelHumidity', 'AvgWindSpeed', 'Humidity',
           'WindDirection', 'WindSpeed', 'Cloudiness'], 1, inplace=True)

# todo: remove midday and evening records

step()

print(w_df[1:11])

# Save sorted values (to see visually which fields lack values the most

# # Reduce messages to one user and take only a part of the data
# user_messages = messages[messages['fromUser.id'] == '55a7c9e08a7b72f55c3f991e']
# user_messages = user_messages[['text', 'sent']]
# processing_size = int(len(user_messages) * (0.01 * processing_percentage))
# user_messages = user_messages[:processing_size]
#
# # Save as one message (it is heavy computation to do it every single time)
# connected_messages = ' '.join(map(str, user_messages.text)).lower()  # lowercase to decrease amount of features
# file = open(f'data/processed_{processing_percentage}.pickle', 'wb')
# pickle.dump(connected_messages, file)
# file.close()
# w_df.to_csv(f'data/processed_{processing_percentage}.csv', index=False)
