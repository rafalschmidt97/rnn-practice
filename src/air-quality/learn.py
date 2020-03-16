import configparser
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import RobustScaler, LabelEncoder


# AQI classification (simplified - https://en.wikipedia.org/wiki/Air_quality_index)
# 0-50 - Good
# 51-100 - Moderate
# 101-150 - Unhealthy for sensitive
# 150+ - Unhealthy

def classify(no2, o3, so2, co):
    value = max(no2, o3, so2, co)
    for index, (min_val, max_val) in enumerate([(0, 50), (51, 100), (101, 150), (151, 500)]):
        if min_val <= value <= max_val:
            return index
    return np.NaN


def prepare(processing):
    # Scale
    pd.options.mode.chained_assignment = None  # disable false warning for copying

    x_transformer = RobustScaler()  # StandardScaler(), MinMaxScalar(feature_range=(-1,1))
    x_transformer = x_transformer.fit(processing[input_features].to_numpy())
    x_scaled = x_transformer.transform(processing[input_features].to_numpy())

    y_encoded = np_utils.to_categorical(processing['Stay'].to_numpy())

    # Shuffle
    sequential_data = []

    for i in range(len(x_scaled) - history_period_size - future_period_predict):
        sequential_data.append([
            x_scaled[i:(i + history_period_size)],
            y_encoded[i + history_period_size + future_period_predict - 1]
        ])

    random.shuffle(sequential_data)

    # Split
    x, y = [], []

    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)

    return np.array(x), np.array(y)


# Loading configuration
config = configparser.ConfigParser()
config.read('config.ini')

processing_percentage = config.getint('APP', 'PROCESSING_PERCENTAGE')
future_period_predict = config.getint('LEARNING', 'FUTURE_PERIOD_PREDICT')
history_period_size = config.getint('LEARNING', 'HISTORY_PERIOD_SIZE')

input_features = ['Year', 'Month', 'Day', 'AvgNO2QI', 'AvgO3AQI', 'AvgSO2AQI', 'AvgCOAQI']

# Toy settings
# batch_size = 32
# epochs = 10
# rnn_layers = [1]
# rnn_node_sizes = [256]
# dense_layers = [1]
# dense_node_sizes = [128]

# Semi Toy settings
batch_size = 32
epochs = 30
rnn_layers = [1, 2]
rnn_node_sizes = [128, 256]
dense_layers = [1, 0, 2]
dense_node_sizes = [128, 256]

# Real settings
# batch_size = 32
# epochs = 30
# rnn_layers = [3, 2, 1]
# rnn_node_sizes = [256, 512, 1024]
# dense_layers = [1, 0, 2]
# dense_node_sizes = [256, 512, 1024, 128, 2048]

# Prepare sequences and predictions
records = pd.read_csv(f'data/processed_small_{processing_percentage}.csv')

# Add target value
records['Stay'] = records[['AvgNO2QI', 'AvgO3AQI', 'AvgSO2AQI', 'AvgCOAQI']] \
    .apply(lambda x: classify(x[0], x[1], x[2], x[3]), axis=1)

# plt.figure(figsize=(10, 6))
# records['Stay'].value_counts().plot.bar()
# plt.legend()
# plt.show()

divider_index = int(len(records) * 0.8)
training_records = records[:divider_index]
validation_records = records[divider_index:]

train_x, train_y = prepare(training_records)
validation_x, validation_y = prepare(validation_records)

# Modeling
for dense_layer in dense_layers:
    for dense_node_index, dense_node_size in enumerate(dense_node_sizes):
        for rnn_layer in rnn_layers:
            for rnn_node_size in rnn_node_sizes:
                if dense_node_index > 0 and dense_layer == 0:
                    continue

                name = f'{rnn_layer}-{rnn_node_size}-rnn-{dense_layer}-{dense_node_size}-dense'
                name = f'{name}-{processing_percentage}-proc-{future_period_predict}-fut-{history_period_size}-his'
                name = f'{name}-{batch_size}-batch-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}'

                print(f'Current computation: {name}')

                model = Sequential()

                model.add(Bidirectional(LSTM(
                    rnn_node_size,
                    input_shape=(train_x.shape[1:]),
                    return_sequences=rnn_layer > 1)
                ))
                model.add(Dropout(0.2))
                model.add(BatchNormalization())

                for layer in range(1, rnn_layer):
                    model.add(Bidirectional(LSTM(rnn_node_size, return_sequences=layer < rnn_layer - 1)))
                    model.add(Dropout(0.2))
                    model.add(BatchNormalization())

                for _ in range(dense_layer):
                    model.add(Dense(dense_node_size))
                    model.add(Dropout(0.2))

                model.add(Dense(train_y.shape[1], activation='softmax'))

                model.compile(
                    loss='categorical_crossentropy',
                    optimizer=Adam(lr=0.001, decay=1e-6),
                    metrics=['accuracy']
                )

                checkpoint = ModelCheckpoint(
                    f'models/{name}_small.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min'
                )
                log_dir = f'logs/{name}'
                tensorboard = TensorBoard(log_dir=log_dir)

                model.fit(
                    train_x, train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(validation_x, validation_y),
                    callbacks=[checkpoint, tensorboard]
                )
