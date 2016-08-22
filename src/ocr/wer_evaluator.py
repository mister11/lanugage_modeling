def get_wer(corr, hyp):
    with open(corr, mode='r') as c:
        c_lines = c.readlines()
    with open(hyp, mode='r') as h:
        h_lines = h.readlines()
    total_wer = 0

    for c_line, h_line in zip(c_lines, h_lines):
        total_wer += wer_for_line(c_line, h_line)

    return 1.0 * total_wer / len(c_lines)

def wer_for_line(c, h):
    c_tokens = c.lower().split()
    h_tokens = h.lower().split()

    diff = 0
    for t in h_tokens:
        if t not in c_tokens or t == 'i' or t == 'u' or t == 'o':
            diff += 1
    return diff

def main():
    correct_file = './test_file_correct.txt'
    hyp_file = './sentences_new_15.txt'
    print(get_wer(correct_file, hyp_file))

if __name__ == '__main__':
    main()
