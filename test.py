import random
import csv

file = open('./wordle.txt', 'r')
words = file.readlines()
words = [word.strip() for word in words]
word_dict = {v:i for i,v in enumerate(words)}
s1 = []
s2 = []
empty = []
for word in words:
    for i in range(15):
        empty = [*word]
        def add():
            global empty
            s1.append(''.join(empty).strip())
            s2.append(word_dict[word])
            empty = [*word]
        for i in range(2):
            empty[random.randint(0, len(empty)-1)] = '_'
        add()
        for i in range(3):
            empty[random.randint(0, len(empty)-1)] = '_'
        add()
        for i in range(1):
            empty[random.randint(0, len(empty)-1)] = '_'
    

writer = csv.writer(open('./wordle.csv', 'w'))
writer.writerow(('obfuscated','labels'))
writer.writerows((zip(s1, s2)))