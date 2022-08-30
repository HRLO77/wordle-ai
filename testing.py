import numpy as np
import pandas as pd
import tensorflow as tf
import pickle as pk
from keras.preprocessing import sequence
import keras

words = [word.strip() for word in open('./wordle.txt').readlines()]
word_dict = {v:i for i,v in enumerate(words)}

dat = pk.load(open('./wordle.pickle', 'rb'))
data = np.array([[i] for i in dat[0]])
labels = np.array([[i] for i in dat[1]])
print(labels[0])

test = np.array([["lowe_"], ["lo_es"]])

test_labels = np.array([['lowes'], ['lowes']])

char2idx = {v:i for i,v in enumerate(set(' '.join(words)))}
# print(char2idx)
idx2char = {v: i for i,v in char2idx.items()}

VOCAB_SIZE = len(char2idx.items())
BATCH_SIZE = 4
MAX_LEN = 250
EPOCHS = 10

model: tf.keras.Sequential = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 64),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.9),
    tf.keras.layers.Dense(len(labels), activation='softmax')
    ])

model.summary()

def encode_text(text):
    # print(text)
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    # print(tokens)
    tokens = [char2idx[word] if word in char2idx else -1 for word in tokens[0]]
    # print(tokens)
    return keras.utils.pad_sequences([tokens], MAX_LEN)[0]


def decode_integers(integers):
    PAD = -1
    text = ""
    for num in integers:
      if num != PAD:
        text += idx2char[num]

    return text[:-1]

def encode_2d(ndarray: np.ndarray):
    l=[]
    for i in ndarray:
        (encode_text("".join(i)))
        l.append(encode_text("".join(i)))
    return keras.utils.pad_sequences(np.array(l), MAX_LEN)

def encode_1d(ndarray: np.ndarray):
    l=[]
    for i in ndarray:
        l.append(encode_text(i))
    return keras.utils.pad_sequences(np.array(l), MAX_LEN)

data: np.ndarray = encode_2d(data)
test: np.ndarray = encode_2d(test)

MAX_LEN=len(words)
labels = encode_2d(labels)

model.compile(loss="binary_crossentropy",optimizer="nadam",metrics=['accuracy'])
history = model.fit(data, labels, epochs=EPOCHS)

model.save('./wordle.h5')

test_labels = encode_2d(test_labels)
MAX_LEN=250

results = model.evaluate(test, test_labels)
test_arr = encode_2d(np.array([["low_s"]]))

def largest(arr: np.ndarray, num: int=3):
    return np.array([i[0] for i in sorted(enumerate(arr), key=lambda x: x[1], reverse=True)[:num]])

def predict():
  result = model.predict(test_arr)
  for i in largest(result):
      print(words[i])

predict()

# test_arr = encode_2d(np.array([["hel"]]))

# predict()

# test_arr = np.array([encode_text("hel")])

# predict()