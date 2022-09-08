# wordle-ai
Trying to make something that solves wordles.

# requirements.txt
Install dependencies with `python -m pip install -r requirements.txt`.
Recommended to have at least 2 free GB ram, 2 core cpu and 1 gb free storage

# dnn_classifier.py
After cloning this repository locally, decompress all the compressed .h5 files. Run the `decompress` function to decompress the .h5 files, and `compress` if you want to compress all the .h5 files again. 

```py
# dnn_classifier.py

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
```
To load up the weights file, use the `build_model` function to build a model and load up weights:

```py
# dnn_classifier.py

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
```
Run `model = build_model(0)` to load the smaller model, or `model = build_model(1)`to load the larger model.

Encode data with the `encode` function (takes a 1d np.ndarray of strings):
```py
# dnn_classifier.py

data = pd.read_csv('./wordle.csv').to_numpy().flatten()

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

del data # remove the extra data
```
And predict using the `predict` function. (returns a tuple of strings and floats, sorted from greatest to least based on the probability):
```py
# dnn_classifier.py

def predict(arr: np.ndarray):
    out = tuple(model.predict(encode(arr.flatten())).flatten())
    out = ((words[i],v) for i,v in enumerate(out))
    return sorted(out, key=lambda x: x[1], reverse=True)
```
# predictions.py
`predictions.py` is an example file of how to encode data, decompress weights files and make predictions.

# diff.py
Run `python diff.py`and follow instructions on the screen, all the work is done automatically, no encoding or training necessary!
