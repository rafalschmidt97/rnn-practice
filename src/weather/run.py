import configparser
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.engine.saving import load_model
from sklearn.preprocessing import RobustScaler

if len(sys.argv) != 3:
    raise Exception('Missing params: model_path processing_percentage')

# Loading configuration
config = configparser.ConfigParser()
config.read('config.ini')

processing_percentage = int(sys.argv[2])
future_period_predict = config.getint('LEARNING', 'FUTURE_PERIOD_PREDICT')
history_period_size = config.getint('LEARNING', 'HISTORY_PERIOD_SIZE')

input_features = ['Year', 'Month', 'Precipitation', 'MaxTemp', 'MinTemp', 'Insolation',
                  'Humidity0', 'Humidity12', 'Humidity18',
                  'Pressure0', 'Pressure12', 'Pressure18',
                  'WindDirection0', 'WindDirection12', 'WindDirection18',
                  'WindSpeed0', 'WindSpeed12', 'WindSpeed18',
                  'Cloudiness0', 'Cloudiness12', 'Cloudiness18']

output_features = ['Precipitation', 'MaxTemp', 'MinTemp']

# Prepare sequences and predictions
all_records = pd.read_csv(f'data/processed_small_{processing_percentage}.csv')
all_records.dropna(inplace=True)  # remove nulls just in case

divider_index = int(len(all_records) * 0.8)
records = all_records[divider_index:]

model_path = sys.argv[1]
model = load_model(model_path)

pd.options.mode.chained_assignment = None  # disable false warning for copying

# Scale
pd.options.mode.chained_assignment = None  # disable false warning for copying

x_transformer = RobustScaler()  # StandardScaler(), MinMaxScalar(feature_range=(-1,1))
x_transformer = x_transformer.fit(all_records[input_features].to_numpy())
x_scaled = x_transformer.transform(records[input_features].to_numpy())

y_transformer = RobustScaler()
y_transformer = y_transformer.fit(all_records[output_features].to_numpy())
y_scaled = y_transformer.transform(records[output_features].to_numpy())

# Prepare sequences
sequential_data = []

for i in range(len(x_scaled) - history_period_size - future_period_predict):
    sequential_data.append([
        x_scaled[i:(i + history_period_size)],
        y_scaled[i + history_period_size + future_period_predict - 1]
    ])

x, y = [], []
for seq, target in sequential_data:
    x.append(seq)
    y.append(target)

# Predict
x_pred = np.array(x)
y_pred = np.array(y)
y_inverse = y_transformer.inverse_transform(y_pred)

predicted = model.predict(x_pred)
predicted_inverse = y_transformer.inverse_transform(predicted)

# print(output_features)
# print('Predicted:')
# print(np.round(predicted_inverse[0], 1))
# print('Original:')
# print(records.iloc[i_pred + history_period_size][output_features].to_numpy())

plt.figure(figsize=(10, 6))
plt.plot(y_inverse[:200, 2], 'b', label='Measured')
plt.plot(predicted_inverse[:200, 2], 'r', label='Predicted')
plt.ylabel('Min Temp (C)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_inverse[:200, 1], 'b', label='Measured')
plt.plot(predicted_inverse[:200, 1], 'r', label='Predicted')
plt.ylabel('Max Temp (C)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_inverse[500:1000, 0], 'b', label='Measured')
plt.plot(predicted_inverse[500:1000, 0], 'r', label='Predicted')
plt.ylabel('Precipitation')
plt.legend()
plt.show()
