# import pandas as pd
# import numpy as np
# import difflib
# import numba
# import os
# prange = numba.prange
# os.environ['NUMBA_OPT'] = 'max'
print('Loading words...')
words = [word.strip() for word in open('./wordle.txt').readlines()]
# possibilities=len(words)
# print('Indexing...')
# idx_dict = {i:v for i,v in enumerate(words)}

# print('Loading dataset...')
# data = pd.read_csv('./wordle.csv')
# print('Numpy-ifying...')
# labels = np.array([int(i) for i in data.pop('labels').to_numpy()], dtype=np.int16).flatten()
# data = data.to_numpy().flatten()
# print('Labelling data...')
# word_dict = {i:idx_dict[v] for i,v in zip(data, labels)}


inputed = input('Enter your word (leave unsure chars _ e.g "_oll_"): ').lower().strip()
invalid = set(input('Enter invalid characters (i.e hgfav): ').lower().strip())
displaced = tuple(input('Enter displaced charcters (i.e hgfav): ').lower().strip())
not_there: list[tuple[str, int]] = []
while len(displaced)!=0:
    comb = input('Enter combinations where displaced letters ARE NOT, diregarding other letters (e.g t____, __t__, _q___). type 11111 to exit: ').lower().strip()
    if comb=='11111':
        break
    if len(comb)!=5 or len(set(comb))!=2:
        print('please enter valid input')
        continue
    for i in range(5):
        if comb[i]!='_':
            break
    not_there.append((comb[i], i))
# @numba.jit(numba.types.Tuple((numba.types.string,))(numba.types.string), boundscheck=False,parallel=True, fastmath=True)
def check(word: str) -> tuple[str]:
    possibles = []
    for possible in words:
        works=True
        for i in range(5):  # 5 letters
            if word[i]=='_':
                if possible[i] in invalid:
                    works = False
                    break  # invalid chars not allowed
            elif word[i]!=possible[i]:
                works = False
                break
        if not works:  # ensure known characters are met
            continue
        met_displaced = 0
        already_displaced = set()
        if len(displaced)!=0:
            for char in displaced:
                for j in range(5):
                    
                    if word[j]=='_' and possible[j]==char and (not (char in already_displaced)) and not_there.count((char, j))==0:
                        met_displaced += 1
                        already_displaced.add(char)
        if met_displaced!=len(displaced):
            continue
        possibles.append(possible)
    return tuple(possibles)

print(check(inputed))
# def remove_dupes(iterable: list):
#     c=0
#     for i in iterable:
#         while i in iterable:
#             iterable.remove(i)
#         iterable.insert(c, i)
#         c+=1
#     return iterable

# def is_likely(word: str):
#     similar_words = []
#     c=0
#     for i in word:
#         if i == inputed[c]:
#             similar_words.append(i)
#         c+=1
#     in_order = []
#     c=0
#     for i in inputed:
#         if not c > len(similar_words)-1:
#             in_order.append(i == similar_words[c])
#             c+=1
#     in_place = []
#     for i in set(displaced):
#         if displaced.count(i) >= word.count(i):
#             in_place.append(True)
#         else:
#             in_place.append(False)
#     if len(similar_words) > 0:
#         return (len(similar_words) >= 3) and not(any(i in invalid for i in word)) and ((sum(in_order) / len(similar_words)) > 0.45) and sum(in_place) == len(in_place)
#     else:
#         return (len(similar_words) >= 3) and not(any(i in invalid for i in word))
    
    
# def score(word: str):
#     similar_words = []
#     c=0
#     for i in word:
#         if i == inputed[c]:
#             similar_words.append(i)
#         c+=1
#     in_order = []
#     c=0
#     for i in inputed:
#         if not c > len(similar_words)-1:
#             in_order.append(i == similar_words[c])
#             c+=1
#     in_place = []
#     for i in set(displaced):
#         if displaced.count(i) >= word.count(i):
#             in_place.append(True)
#         else:
#             in_place.append(False)
#     if len(similar_words) > 0:
#         return (len(similar_words) / 5) + (sum(in_order) / len(similar_words)) + (sum(in_place) / (len(displaced) if len(displaced)>0 else 1))
#     else:
#         return (len(similar_words) / 5)


# c=1
# print('Finding matches...')
# out = sorted(((score(word_dict[i]), i) for i in remove_dupes(difflib.get_close_matches(inputed, data, n=possibilities, cutoff=0.8))[:100]), key=lambda x: x[0], reverse=True)
# print('Done!')
# remade = []
# to_add = []
# for i,v in out:
#     if is_likely(word_dict[v]):
#         remade.append(f'{word_dict[v]} - score: {i}')
#         c+=1
#     elif is_likely(v):
#         to_add.append(f'{word_dict[v]} - score: {i} (unsure)')
#         c+=1
# print(*remade+to_add,sep='\n')
