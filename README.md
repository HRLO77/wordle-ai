# wordle-ai
Trying to make something that solves wordles.

# dnn_classifier.py
To load up the weights file, create a keras.Sequential model such as
```py
model = keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(5,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(len(set(labels)), activation='softmax')
  ])

model.compile(optimizer='nadam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.load_weights('./smaller_model.h5')
```
For the smaller model, otherwise:
```py
model = keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(5,)),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dense(len(set(labels)), activation='softmax')
  ])

model.compile(optimizer='nadam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.load_weights('./larger_model.h5')
```
To load the larger model.

Encode data with:
```py
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
And predict using (returns a tuple of strings and floats, sorted from greatest to least based on the probability):
```py
def predict(arr: np.ndarray):
    out = tuple(model.predict(encode(arr.flatten())).flatten())
    out = ((words[i],v) for i,v in enumerate(out))
    return sorted(out, key=lambda x: x[1], reverse=True)
```
# diff.py
Run `python diff.py`and follow instructions on the screen, all the work is done automatically, no encoding or training necessary!
