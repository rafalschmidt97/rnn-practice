import configparser
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.engine.saving import load_model
from keras.utils import np_utils
from sklearn.preprocessing import RobustScaler


def classify(no2, o3, so2, co):
    value = max(no2, o3, so2, co)
    for index, (min_val, max_val) in enumerate([(0, 50), (51, 100), (101, 150), (151, 500)]):
        if min_val <= value <= max_val:
            return index
    return np.NaN


if len(sys.argv) != 3:
    raise Exception('Missing params: model_path processing_percentage')

# Loading configuration
config = configparser.ConfigParser()
config.read('config.ini')

processing_percentage = int(sys.argv[2])
future_period_predict = config.getint('LEARNING', 'FUTURE_PERIOD_PREDICT')
history_period_size = config.getint('LEARNING', 'HISTORY_PERIOD_SIZE')

input_features = ['Year', 'Month', 'Day', 'AvgNO2QI', 'AvgO3AQI', 'AvgSO2AQI', 'AvgCOAQI']

# Prepare sequences and predictions
records = pd.read_csv(f'data/processed_small_{processing_percentage}.csv')

records['Stay'] = records[['AvgNO2QI', 'AvgO3AQI', 'AvgSO2AQI', 'AvgCOAQI']] \
    .apply(lambda x: classify(x[0], x[1], x[2], x[3]), axis=1)

model_path = sys.argv[1]
model = load_model(model_path)

# Scale
pd.options.mode.chained_assignment = None  # disable false warning for copying

x_transformer = RobustScaler()
x_transformer = x_transformer.fit(records[input_features].to_numpy())
x_scaled = x_transformer.transform(records[input_features].to_numpy())

y_encoded = np_utils.to_categorical(records['Stay'].to_numpy())
y_labels = ['Good', 'Moderate', 'Quite unhealthy', 'Unhealthy']

# Prepare sequences
sequential_data = []

for i in range(len(x_scaled) - history_period_size - future_period_predict):
    sequential_data.append([
        x_scaled[i:(i + history_period_size)],
        y_encoded[i + history_period_size + future_period_predict - 1]
    ])

x, y = [], []
for seq, target in sequential_data:
    x.append(seq)
    y.append(target)

# Predict
x_pred = np.array(x)
y_pred = np.array(y)
y_inverse = y_pred.argmax(axis=1)

predicted = model.predict(x_pred)
predicted_inverse = np.argmax(predicted, axis=1)

predicted_inverse_labeled = [y_labels[i] for i in predicted_inverse]
y_inverse_labeled = [y_labels[i] for i in y_inverse]

df = pd.DataFrame(np.column_stack([y_inverse_labeled, predicted_inverse_labeled]), columns=['Dataset', 'Predicted'])
pd.set_option('display.max_rows', None)
print(df[530:545])

plt.figure(figsize=(12, 6))
plt.plot(y_inverse[530:545], 'g', alpha=0.7, label='Measured')
plt.plot(predicted_inverse[530:545], 'b:', label='Predicted')
plt.ylabel('Prediction')
plt.legend()
plt.show()
