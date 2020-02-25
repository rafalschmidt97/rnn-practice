import configparser
import pickle

import matplotlib.pyplot as plt
import pandas as pd

# Loading configuration
config = configparser.ConfigParser()
config.read('config.ini')

processing_percentage = config.getint('APP', 'PROCESSING_PERCENTAGE')

# Selecting data
messages = pd.read_csv('data/freecodecamp_casual_chatroom.csv',
                       usecols=['fromUser.id', 'fromUser.displayName', 'text', 'sent'])
users_to_ignore = ['55b977f00fc9f982beab7883']  # ignore bots (CamperBot)
messages = messages[~messages['fromUser.id'].isin(users_to_ignore)]

# Show a chart with the most active users
plt.style.use('fivethirtyeight')
_, axis = plt.subplots(figsize=(12, 9))
messages['fromUser.id'].value_counts().head(5).plot.bar(color='green')
axis.set_xticklabels(axis.get_xticklabels(), rotation=25)
plt.title('Most active users')
plt.show()
print(messages['fromUser.id'].value_counts().head(5))  # the most 55a7c9e08a7b72f55c3f991e - 141362

# Reduce messages to one user and take only a part of the data
user_messages = messages[messages['fromUser.id'] == '55a7c9e08a7b72f55c3f991e']
user_messages = user_messages[['text', 'sent']]
processing_size = int(len(user_messages) * (0.01 * processing_percentage))
user_messages = user_messages[:processing_size]

# Save as one message (it is heavy computation to do it every single time)
connected_messages = ' '.join(map(str, user_messages.text)).lower()  # lowercase to decrease amount of features
file = open(f'data/processed_{processing_percentage}.pickle', 'wb')
pickle.dump(connected_messages, file)
file.close()
