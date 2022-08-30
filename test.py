import numpy as np
import pickle as pk
import random

file = open('./wordle.txt', 'r')
words = file.readlines()
words = [word.strip() for word in words]
word_dict = {v:i for i,v in enumerate(words)}
s1 = []
s2 = []
empty = []
for word in words:
    for i in range(3):
        empty = [*word]
        for i in range(len(empty)//2):
            empty[random.randint(0, len(empty)-1)] = '_'
        s1.append(''.join(empty).strip())
        s2.append(word)
with open('./wordle.pickle', 'wb') as f:
    pk.dump(np.array([s1, s2]), f)