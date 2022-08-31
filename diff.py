import pandas as pd
import numpy as np
import difflib

print('Loading words...')
words = [word.strip() for word in open('./wordle.txt').readlines()]
possibilities=len(words)
print('Indexing...')
idx_dict = {i:v for i,v in enumerate(words)}

print('Loading dataset...')
data = pd.read_csv('./wordle.csv')
print('Numpy-ifying...')
labels = np.array([int(i) for i in data.pop('labels').to_numpy()], dtype=np.int16).flatten()
data = data.to_numpy().flatten()
print('Labelling data...')
word_dict = {i:idx_dict[v] for i,v in zip(data, labels)}


inputed = input('Enter your word: ')
invalid = set(input('Enter invalid characters (i.e hgfav): '))

def remove_dupes(iterable: list):
    c=0
    for i in iterable:
        while i in iterable:
            iterable.remove(i)
        iterable.insert(c, i)
        c+=1
    return iterable

def is_likely(word: str):
    similar_words = []
    c=0
    for i in word:
        if i in invalid:
            return False
        if i == inputed[c]:
            similar_words.append(i)
        c+=1
    return len(similar_words) > 3
    
def score(word: str):
    similar_words = []
    c=0
    for i in word:
        if i == inputed[c]:
            similar_words.append(i)
        c+=1
    return (len(similar_words) / 5)


c=1
print('Finding matches...')
out = sorted(((score(i), i) for i in remove_dupes(difflib.get_close_matches(inputed, data, n=possibilities, cutoff=0.7))[:100]), key=lambda x: x[0], reverse=True)
print('Done!')
for i,v in out:
    if is_likely(v):
        print(f'{c}. {word_dict[v]} - score: {i}')
        c+=1