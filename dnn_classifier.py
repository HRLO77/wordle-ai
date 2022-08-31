import numpy as np
import pandas as pd
import tensorflow as tf
import pickle as pk
from keras.preprocessing import sequence
import keras

words = [word.strip() for word in open('./wordle.txt').readlines()]
word_dict = {v:i for i,v in enumerate(words)}


data = pd.read_csv('./wordle.csv')
labels = np.array([int(i) for i in data.pop('labels').to_numpy()])
test =pd.read_csv('./test.csv')
test_labels = np.array([int(i) for i in test.pop('labels').to_numpy()])
test.head(), data.head()
char2idx = {v:i for i,v in enumerate(set(' '.join(words)))}
idx2char = {v:i for i,v in char2idx.items()}

def train_fn(features, labels, training=True, batch_size=1):
    """An input function for training or evaluating"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

my_feature_columns = []
for key in data.keys():
    my_feature_columns.append(tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket(key=key, hash_bucket_size=30)))
    
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=len(set(words)))


classifier.train(
    input_fn=lambda: train_fn(data, labels, training=True),
    steps=10000)

eval_result = classifier.evaluate(
    input_fn=lambda: train_fn(test, test_labels, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

def input_fn(features, batch_size=1):
    """An input function for prediction."""
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


predict_x = {
    'obfuscated': ["lo_e_", 'fle_s']
}


predictions = classifier.predict(
    input_fn=lambda: input_fn(predict_x))

print(classifier.predict(input_fn=lambda: input_fn(predict_x))[0]['class_ids'][0])

for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    print(pred_dict['class_ids'])
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        words[class_id], 100 * probability))
    
while True:
    exec(input(), globals(), locals())