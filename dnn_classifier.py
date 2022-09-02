print('Importing libraries...')
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle as pk
from keras.preprocessing import sequence
import keras

print('Loading data...')
words = [word.strip() for word in open('./wordle.txt').readlines()]
word_dict = {v:i for i,v in enumerate(words)}
idx_dict = {i:v for i,v in enumerate(words)}

print('Indexing and labelling...')
data = pd.read_csv('./wordle.csv')
labels = np.array([int(i) for i in data.pop('labels').to_numpy(dtype=np.int16)])
test =pd.read_csv('./test.csv')
test_labels = np.array([int(i) for i in test.pop('labels').to_numpy(dtype=np.int8)])
test.head(), data.head()

data = data.to_numpy().flatten()
test = test.to_numpy().flatten()

t = sorted(set(''.join(data)))
char2idx = {v:i for i,v in enumerate(t)}
idx2char = {i:v for i,v in enumerate(t)}
del t

print('Test:')
print(data[0], labels[0])
print(idx_dict[labels[0]])

def encode(array: np.ndarray):
    l = []
    for i in array:
        l.append([char2idx[v] for v in i])
    return np.array(l, dtype=np.int8)

data = encode(data)
test = encode(test)

data = data / data.max()

test = test / test.max()

model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(5,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(set(labels)), activation='softmax')
    ])

model.compile(optimizer='nadam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.summary()

model.fit(data, labels, epochs=3)

model.save('./model.h5')

history = model.evaluate(test, test_labels)

print(history)


def predict(arr: np.ndarray):
    out: np.ndarray = model.predict(arr).flatten()
    m = (out.min(), out.min())

    c=0
    for i in out:
        if i > m[1]:
            min = (i,c)
        c+=1

    print(idx_dict[m[1]], m[0])
    return m        