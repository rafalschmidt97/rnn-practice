import configparser
import pickle
import random
from datetime import datetime

import numpy as np
from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.optimizers import Adam


def preprocess(processing_messages):
    sequences = []
    for i in range(0, len(processing_messages) - sentence_length, overlapping_step):
        seq_data = processing_messages[i: i + sentence_length]
        seq_pred = processing_messages[i + sentence_length]
        sequences.append([seq_data, seq_pred])

    random.shuffle(sequences)

    x = np.zeros((len(sequences), sentence_length, len(characters)), dtype=np.bool)
    y = np.zeros((len(sequences), len(characters)), dtype=np.bool)

    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence[0]):
            x[i, t, character_map[char]] = 1
        y[i, character_map[sequence[1]]] = 1
    return x, y


# Loading configuration
config = configparser.ConfigParser()
config.read('config.ini')

processing_percentage = config.getint('APP', 'PROCESSING_PERCENTAGE')
sentence_length = config.getint('LEARNING', 'SENTENCE_LENGTH')

# Toy settings
# overlapping_step = 10
# batch_size = 128
# epochs = 10
# rnn_layers = [1]
# rnn_node_sizes = [512]
# dense_layers = [1]
# dense_node_sizes = [256]

# Real settings
overlapping_step = 10
batch_size = 128
epochs = 25
rnn_layers = [1, 2]
rnn_node_sizes = [256, 512]
dense_layers = [1, 2]
dense_node_sizes = [256, 512]

# Prepare sequences and predictions
file = open(f'data/processed_{processing_percentage}.pickle', 'rb')
connected_messages = pickle.load(file)
characters = sorted(list(set(connected_messages)))
character_map = dict((c, i) for i, c in enumerate(characters))
indicator_map = dict((i, c) for i, c in enumerate(characters))

divider_index = int(len(connected_messages) * 0.8)
training_messages = connected_messages[:divider_index]
validation_messages = connected_messages[divider_index:]

train_x, train_y = preprocess(training_messages)
validation_x, validation_y = preprocess(validation_messages)

# Modeling
for dense_layer in dense_layers:
    for dense_node_index, dense_node_size in enumerate(dense_node_sizes):
        for rnn_layer in rnn_layers:
            for rnn_node_size in rnn_node_sizes:
                if dense_node_index > 0 and dense_layer == 0:
                    continue

                name = f'{rnn_layer}-{rnn_node_size}-rnn-{dense_layer}-{dense_node_size}-dense'
                name = f'{name}-{processing_percentage}-proc-{sentence_length}-len-{overlapping_step}-lap'
                name = f'{name}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}'

                print(f'Current computation: {name}')

                model = Sequential()

                model.add(LSTM(
                    rnn_node_size,
                    input_shape=(sentence_length, len(characters)),
                    return_sequences=rnn_layer > 1)
                )
                model.add(Dropout(0.2))
                model.add(BatchNormalization())

                for layer in range(1, rnn_layer):
                    model.add(LSTM(rnn_node_size, return_sequences=layer < rnn_layer - 1))
                    model.add(Dropout(0.1))
                    model.add(BatchNormalization())

                for _ in range(dense_layer):
                    model.add(Dense(dense_node_size, activation='relu'))
                    model.add(Dropout(0.2))

                model.add(Dense(len(characters), activation='softmax'))

                model.compile(
                    loss='categorical_crossentropy',
                    optimizer=Adam(lr=0.001, decay=1e-6),  # RMSprop(lr=0.01)
                    metrics=['accuracy']
                )

                checkpoint = ModelCheckpoint(
                    f'models/{name}.h5',
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
