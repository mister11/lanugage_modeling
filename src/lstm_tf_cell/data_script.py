import cPickle as pickle
from os import listdir
from collections import Counter

dataset_files_path = '/home/svidak/dr/1-billion/'
files = listdir(dataset_files_path + 'text/')

def create_dict_count():
    dict_count = Counter()

    for filename in files:
        print(filename)
        with open(dataset_files_path + "text/" + filename, mode='r') as f:
            for line in f.readlines():
                tokens = line.strip().lower().split()
                dict_count.update(tokens)
                dict_count.update([' '] * (len(tokens) - 1))

    dict_count = {word: freq
                  for word, freq in dict_count.items() if freq >= 100}
    pickle.dump(dict_count, open(dataset_files_path + "word_dict_count.ser",
                                 mode='wb'))


def word_dict_count():
    c = Counter()
    files = listdir(dataset_files_path + 'text/chunks/train')
    files = ['/'.join([dataset_files_path + 'text/chunks/train', file])
             for file in files]
    for file in files:
        print(file)
        with open(file, mode='r') as f:
            for line in f:
                tokens = line.strip().lower().split()
                c.update(tokens)
    c = Counter({k: v for k, v in c.items() if v >= 1000})
    pickle.dump(c, open(dataset_files_path + 'word_dict_count_nosp.pkl', mode='wb'))


def char_dict_count():
    c = Counter()
    files = listdir(dataset_files_path + 'text/chunks/train')
    files = ['/'.join([dataset_files_path + 'text/chunks/train', file])
             for file in files]
    for file in files:
	print(file)
        with open(file, mode='r') as f:
            for line in f:
                process_line(c, unicode(line, 'utf-8'))
    pickle.dump(c, open(dataset_files_path + 'char_count_dict.pkl', mode='wb'))

def process_line(c, line):
    line = line.strip().lower()
    c.update(list(line))

def merge_texts():
    with open(dataset_files_path + "text/all_texts.txt", mode='w') as out:
        for filename in files:
            print(filename)
            with open(dataset_files_path + 'text/' + filename, mode='r') as f:
                for line in f.readlines():
                    out.write(line)


def split_data():
    with open(dataset_files_path + "text/all_texts.txt", mode='r') as f:
        lines = f.readlines()
        size = len(lines)
        train_data_size = (int)(0.7 * size)
        valid_data_size = test_data_size = (int)(0.15 * size)

        with open(dataset_files_path + "text/train_data.txt", mode='w') as tr:
            with open(dataset_files_path +
                      "text/valid_data.txt", mode='w') as va:
                with open(dataset_files_path +
                          "text/test_data.txt", mode='w') as te:
                    for line in lines[:train_data_size]:
                        tr.write(line)
                    for line in lines[train_data_size:train_data_size +
                                      valid_data_size]:
                        va.write(line)
                    for line in lines[train_data_size + valid_data_size:]:
                        te.write(line)

def split_hrwac():
	file_len = 500000
	index = 0
	out_file_name_template = dataset_files_path + 'text/chunks/part_'
	part = 1
	out_file = open(out_file_name_template + str(part), mode='w')
	with open(dataset_files_path + 'text/all_texts.txt') as f:
		for line in f:
			if index == file_len:
				out_file.close()
				part += 1
				index = 0
				out_file = open(out_file_name_template + str(part), mode='w')
			out_file.write(line)
			index += 1

if __name__ == '__main__':
    char_dict_count()
