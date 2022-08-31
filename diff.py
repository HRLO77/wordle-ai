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
print(invalid)

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
        if i == inputed[c]:
            similar_words.append(i)
        c+=1
    in_order = []
    c=0
    for i in inputed:
        if not c > len(similar_words)-1:
            in_order.append(i == similar_words[c])
            c+=1
    if len(similar_words) > 0:
        return (len(similar_words) >= 3) and not(any(i in invalid for i in word)) and ((sum(in_order) / len(similar_words)) > 0.45)
    else:
        return (len(similar_words) >= 3) and not(any(i in invalid for i in word))
    
    
def score(word: str):
    similar_words = []
    c=0
    for i in word:
        if i == inputed[c]:
            similar_words.append(i)
        c+=1
    in_order = []
    c=0
    for i in inputed:
        if not c > len(similar_words)-1:
            in_order.append(i == similar_words[c])
            c+=1
    if len(similar_words) > 0:
        return (len(similar_words) / 5) + (sum(in_order) / len(similar_words))
    else:
        return (len(similar_words) / 5)


c=1
print('Finding matches...')
out = sorted(((score(word_dict[i]), i) for i in remove_dupes(difflib.get_close_matches(inputed, data, n=possibilities, cutoff=0.1))[:100]), key=lambda x: x[0], reverse=True)
print('Done!')
remade = []
to_add = []
for i,v in out:
    if is_likely(word_dict[v]):
        remade.append(f'{word_dict[v]} - score: {i}')
        c+=1
    elif is_likely(v):
        to_add.append(f'{word_dict[v]} - score: {i} (unsure)')
        c+=1
print(*remade+to_add,sep='\n')