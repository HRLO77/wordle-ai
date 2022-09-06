print('Importing libraries...')
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import gzip
import glob


def compress():
    '''Compress data in all .h5 files'''
    for i in glob.glob('./*.h5'):
        l = gzip.compress(open(i, 'rb').read(),compresslevel=9)
        open(i, 'wb').write(l)
        
def decompress():
    '''Decompress data in all .h5 files'''
    for i in glob.glob('./*.h5'):
        l = gzip.decompress(open(i, 'rb').read())
        open(i, 'wb').write(l)

print('Decompressing weights files...')
decompress() # remove this line if .h5 files are already decompressed

print('Loading data...')
words = [word.strip() for word in open('./wordle.txt').readlines()]
word_dict = {v:i for i,v in enumerate(words)}
idx_dict = {i:v for i,v in enumerate(words)}

print('Indexing and labelling...')
data = pd.read_csv('./wordle.csv')
labels = np.array([int(i) for i in data.pop('labels').to_numpy(dtype=np.int16)])
test =pd.read_csv('./test.csv')
test_labels = np.array([int(i) for i in test.pop('labels').to_numpy(dtype=np.int16)])
test.head(), data.head()

data = data.to_numpy().flatten()
test = test.to_numpy().flatten()

t = sorted(set(''.join(data)))
char2idx = {v:i for i,v in enumerate(t)}
idx2char = {i:v for i,v in enumerate(t)}
del t

def encode(array: np.ndarray):
    l = []
    for i in array:
        l.append([char2idx[v] for v in i])
    d = np.array(l, dtype=np.int8)
    return d

print('Encoding data...')
data = encode(data)
test = encode(test)

model = keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(5,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(len(set(labels)), activation='softmax')
  ])

model.compile(optimizer='nadam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'], jit_compile=True)


def build_model(mode: int=0):
    '''Returns a keras.Sequential model with the loaded weights.'''
    mode+=1
    model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(5,)),
    tf.keras.layers.Dense(128*mode, activation='relu'),
    tf.keras.layers.Dense(256*mode, activation='relu'),
    tf.keras.layers.Dense(512*mode, activation='relu'),
    tf.keras.layers.Dense(len(set(labels)), activation='softmax')
    ])
    model.compile(optimizer='nadam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'], jit_compile=True)
    model.summary()
    model.load_weights('./smaller_model.h5' if mode==1 else './larger_model.h5')
    return model

model.fit(data, labels, epochs=25)
model.save('./new_weights.h5')
history = model.evaluate(test, test_labels)

def predict(arr: np.ndarray):
    out = tuple(model.predict(encode(arr.flatten())).flatten())
    out = ((words[i],v) for i,v in enumerate(out))
    return sorted(out, key=lambda x: x[1], reverse=True)


# test lines to showcase accuracy
# outputs tuple[tuple[str, float]]
print(*predict(np.array(['fle_s']))[:5])
print(*predict(np.array(['l_wes']))[:5])
print(*predict(np.array(['se_se']))[:5])
print(*predict(np.array(['s__se']))[:5])
print(*predict(np.array(['w_t_r']))[:5])
# the [:5] takes the 5 most likely options
