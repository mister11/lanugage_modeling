from collections import Counter
import cPickle as pickle

dict_count = Counter()

with open("/home/svidak/git/text/all_texts_2.txt", mode='r') as f:
    for i, line in enumerate(f.readlines()):
        if i % 1000000 == 0:
            print(i)
        tokens = line.strip().lower().split()
        dict_count.update(tokens)
        dict_count.update([' '] * (len(tokens) - 1))

dict_count = {word: freq for word, freq in dict_count.items() if freq >= 1000}
pickle.dump(dict_count, open("/home/svidak/git/word_dict_count.ser", mode='wb'))
