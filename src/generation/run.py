import configparser
import pickle
import random
import sys

import numpy as np
from keras.engine.saving import load_model


def sample_index(prediction, temperature=1.0):
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / temperature
    exp_prediction = np.exp(prediction)
    prediction = exp_prediction / np.sum(exp_prediction)
    prediction = np.random.multinomial(1, prediction, 1)
    return int(np.argmax(prediction))


if len(sys.argv) != 3:
    raise Exception('Missing params: model_path processing_percentage')

# Loading configuration
config = configparser.ConfigParser()
config.read('config.ini')

processing_percentage = int(sys.argv[2])
sentence_length = config.getint('LEARNING', 'SENTENCE_LENGTH')

# Prepare sequences and predictions
file = open(f'data/processed_{processing_percentage}.pickle', 'rb')
connected_messages = pickle.load(file)
characters = sorted(list(set(connected_messages)))
character_map = dict((c, i) for i, c in enumerate(characters))
indicator_map = dict((i, c) for i, c in enumerate(characters))

model_path = sys.argv[1]
model = load_model(model_path)

generated_fragment_length = 400
sentence_starting_point = random.randint(0, len(connected_messages) - sentence_length - 1 - generated_fragment_length)
sentence = connected_messages[sentence_starting_point: sentence_starting_point + sentence_length]
print(f'Used fragment: {sentence}')

for diversity in [0.2, 0.5, 1.0, 1.2]:
    print(f'Diversity: {diversity}')
    moving_sentence = '' + sentence
    generated_sentence = '' + sentence
    for i in range(400):
        x_pred = np.zeros((1, sentence_length, len(characters)))
        for index, char in enumerate(moving_sentence):
            x_pred[0, index, character_map[char]] = 1.

        predicted = model.predict(x_pred)[0]
        predicted_index = sample_index(predicted, diversity)
        predicted_character = indicator_map[predicted_index]

        moving_sentence = moving_sentence[1:] + predicted_character
        generated_sentence += predicted_character
    print(generated_sentence)

print("Original long:")
print(connected_messages[sentence_starting_point: sentence_starting_point + generated_fragment_length])
