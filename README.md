# wordle-ai
Trying to make something that solves wordles.


To load up the weights file, create a keras.Sequential model such as
```py
model = keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(5,)),
  tf.keras.layers.Dense(64*4*4, activation='relu'),
  tf.keras.layers.Dense(64*4, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(len(set(labels)), activation='softmax')
  ])

model.compile(optimizer='nadam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
```
