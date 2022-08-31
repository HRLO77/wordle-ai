import pandas as pd
import numpy as np
import difflib
words = [word.strip() for word in open('./wordle.txt').readlines()]
possibilities=len(words)

data = pd.read_csv('./wordle.csv')
labels = np.array([int(i) for i in data.pop('labels').to_numpy()], dtype=np.int16).flatten()
data = data.to_numpy().flatten()
word_dict = {i:v for i,v in zip(data, labels)}


input = input('Enter your word: ')

for i in difflib.get_close_matches(input, data, n=3, cutoff=0.5):
    print(words[word_dict[i]])