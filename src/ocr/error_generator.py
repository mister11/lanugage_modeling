import argparse
import random
from os import listdir

def generate_errors(infile, outfile, args):
    with open(infile, mode='r') as inf:
        with open(outfile, mode='w') as outf:
            for line in inf:
                outf.write(generate_error(line, args.error_rate) + "\n")

def generate_error(line, err_rate):
    old_tokens = line.split()
    new_tokens = []
    for token in old_tokens:
        if len(new_tokens) == 0:
            new_tokens.append(token)
            continue
        if random.random() < err_rate:
            current = new_tokens[-1]
            new_tokens.pop()
            new_tokens.append(current + token)
        else:
            new_tokens.append(token)
    return " ".join(new_tokens)

def main(args):
    indir = './data/correct/'
    outdir = './data/error/'
    for infile in listdir(indir):
        generate_errors(indir + infile, outdir + infile, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-er', '--error_rate',
                        help="percentage of error occuring in one sentence",
                        type=float,
                        default=0.1)
    main(parser.parse_args())
