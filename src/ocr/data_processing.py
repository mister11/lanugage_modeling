from os import listdir
from collections import Counter
import cPickle as pickle

data_dir = '/home/svidak/dr/ocr/'
train_size = 0.6

def ocr_vocab_count():
    counter = Counter()
    files = sorted(listdir(data_dir + 'correct/train'))
    for file in files:
        filename = '/'.join([data_dir, 'correct/train', file])
        print(filename)
        with open(filename, mode='r') as f:
            for line in f:
                tokens = line.strip().lower().split()
                counter.update(tokens)
    pickle.dump(counter, open(data_dir + 'ocr_vocab_count.pkl', mode='wb'))

def merge_vocabs(save_file, ocr_vocab_count_threshold=1):
    hrwac = pickle.load(open(data_dir + 'word_dict_count_nosp_3000.pkl', mode='rb'))
    ocr = pickle.load(open(data_dir + 'ocr_vocab_count.pkl', mode='rb'))

    total = hrwac.copy()
    total.update({k: v for k, v in ocr.items()
                  if v >= ocr_vocab_count_threshold})

    pickle.dump(total, open(save_file + '_' +
                            str(ocr_vocab_count_threshold) + '.pkl', mode='wb'))

if __name__ == '__main__':
    ocr_vocab_count()
    merge_vocabs(data_dir + '/ocr_final_vocab_thresh', 1)
