"""
An implementation of Gale-Church algorithm in Python 3
"""

import math
import time
import argparse
from os import listdir, remove
from os.path import isfile, join, basename
from file_read_back.file_read_back import FileReadBackwards
import multiprocessing as mp
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--corpus-folder-source', '-src')
parser.add_argument('--corpus-folder-target', '-trg')
parser.add_argument('--output-folder', '-out')
parser.add_argument('--load-to-memory-threshold', '-ltmr', default=1000)
parser.add_argument('--score-cutoff', '-cutoff', default=10000000)
parser.add_argument('--alignment-type', '-align', choices=['pharaoh', 'text'], default='pharaoh')
parser.add_argument('--num-proc', '-proc', default=0)
args = parser.parse_args()

# Alignment costs: -100*log(p(x:y)/p(1:1))
bead_costs = {
     (2, 2): 440,
     (2, 1): 230,
     (1, 2): 230,
     (1, 1): 0,
     (1, 0): 450,
     (0, 1): 450
}

# Length cost parameters
mean_xy = 1
variance_xy = 6.8
LOG2 = math.log(2)


def norm_cdf(z):
    """ Cumulative distribution for N(0, 1) """
    t = 1 / (1 + 0.2316419 * z)
    return (1 - 0.3989423 * math.exp(-z * z / 2) *
            ((((1.330274429 * t - 1.821255978) * t
               + 1.781477937) * t - 0.356563782) * t + 0.319381530) * t)


def norm_logsf(z):
    """ Logarithm of the survival function for N(0, 1) """
    try:
        return math.log(1 - norm_cdf(z))
    except ValueError:
        return float('-inf')


def length_cost(sx, sy):
    """ -100*log[p(|N(0, 1)|>delta)] """
    lx, ly = sum(sx), sum(sy)
    m = (lx + ly * mean_xy) / 2
    try:
        delta = (lx - ly * mean_xy) / math.sqrt(m * variance_xy)
    except ZeroDivisionError:
        return float('-inf')
    return - 100 * (LOG2 + norm_logsf(abs(delta)))


def calc_cost(i, j, x, y, m):
    costs = []
    if i == j == 0:
        return (0,0,0)
    else:
        costs.append(min((m[i - di, j - dj][0] + length_cost(x[i - di:i], y[j - dj:j]) + bead_cost, di, dj) for (di, dj), bead_cost in bead_costs.items() if i - di >= 0 and j - dj >= 0))
    return min(costs)


def calc_cost_large(i, j, x, y, scorecutoff, m):
    costs = []
    if i == j == 0:
        return (0,0,0)
    else:
        try:
            costs.append(min((m[i - di][j - dj][0] + length_cost(x[i - di:i], y[j - dj:j]) + bead_cost, di, dj) for (di, dj), bead_cost in bead_costs.items() if i - di >= 0 and j - dj >= 0))
        except Exception as e:
            print(e)
            print(scorecutoff, i, j)
            return (scorecutoff, i, j)
    return min(costs)


def _align(x, y, longlength=1000, scorecutoff=None, my_basename='filename'):
    m = {}
    max_length = max(len(x), len(y))
    temp = tempfile.NamedTemporaryFile(prefix='gc_' + my_basename + '_', delete=False, mode='w')
    highest_score = 0

    if max_length > longlength:
        for i in range(len(x)+1):
            m[i] = {}
            for j in range(len(y)+1):
                min_i_j = calc_cost_large(i, j, x, y, scorecutoff, m)
                m[i][j] = min_i_j

            for key, value in m[i].items():
                if not math.isinf(value[0]):
                    if scorecutoff is None:
                        temp.write(str(key) + '|' + str(int(value[0])) + '|' + str(value[1]) + '|' + str(value[2]) + '\t')
                    elif int(value[0]) < scorecutoff:
                        temp.write(str(key) + '|' + str(int(value[0])) + '|' + str(value[1]) + '|' + str(value[2]) + '\t')
            temp.write('\n')
            if i - 3 >= 0:
                m[i - 3] = {}
    temp.close()

    if max_length > longlength:
        i, j = len(x), len(y)
        with FileReadBackwards(temp.name) as fi:
            tempdict = {}
            currLine = i
            for line in fi:
                di = 0
                if currLine == i:
                    currvalues = line.strip().split('\t')
                    for cv in currvalues:
                        v = cv.split('|')
                        try:
                            tempdict[int(v[0])] = (int(v[1]), int(v[2]), int(v[3]))
                        except: #length cost was too large for all combinations
                            tempdict[0] = (100000000, 1, 0)
                    while di == 0:
                        try:
                            (c, di, dj) = tempdict[j]
                        except: #Exception for the case of length cost having been cut off when generating the temp file
                            (c, di, dj) = (100000000, 0, 1)
                        if c > highest_score:
                            highest_score = c
                        if di == dj == 0:
                            break
                        yield (i - di, i), (j - dj, j), highest_score, len(x), len(y)
                        i -= di
                        j -= dj
                    if di <= dj <= 0:
                        break

                currLine -= 1
    else:
        for i in range(len(x)+1):
            for j in range(len(y)+1):
                m[i, j] = calc_cost(i, j, x, y, m)

        while True:
            (c, di, dj) = m[i, j]
            if c > highest_score:
                highest_score = c
            if di == dj == 0:
                break
            yield (i-di, i), (j-dj, j), highest_score, len(x), len(y)
            i -= di
            j -= dj
    remove(temp.name)


def char_length(sentence):
    """ Length of a sentence in characters """
    return len(sentence.replace(' ', ''))


def align(sx, sy, longlength, scorecutoff, my_basename):
    """ Align two groups of sentences """
    cx = list(map(char_length, sx))
    cy = list(map(char_length, sy))
    for (i1, i2), (j1, j2), highest_score, len_x, len_y in reversed(list(_align(cx, cy, longlength, scorecutoff, my_basename))):
        source_sentences = range(i1, i2)
        target_sentences = range(j1, j2)
        if args.alignment_type == 'pharaoh':
            yield str(list(source_sentences)) + ':' + str(list(target_sentences)), highest_score, len_x, len_y
        elif args.alignment_type == 'text':
            yield ' '.join(sx[i1:i2]), ' '.join(sy[j1:j2]), highest_score, len_x, len_y


def read_blocks(f):
    # Blocks are separated by an empty line. They can be paragraphs or documents.
    block = []
    for l in f:
        if not l.strip():
            yield block
            block = []
        else:
            block.append(l.strip())
    if block:
        yield block


def main(corpus_x, corpus_y, longlength, scorecutoff):
    alignments_out = ''
    with open(corpus_x) as fx, open(corpus_y) as fy:
        for block_x, block_y in zip(read_blocks(fx), read_blocks(fy)):
            for alignment, highest_score, len_x, len_y in align(block_x, block_y, longlength, scorecutoff, basename(corpus_x)):
                alignments_out += alignment + '\n'
    return alignments_out, highest_score, len_x, len_y


def main_multi(filename):
    startTime = time.time()
    longlength = int(args.load_to_memory_threshold)
    scorecutoff = None
    if args.score_cutoff is not None:
        scorecutoff = int(args.score_cutoff)
    with open(args.output_folder.rstrip('/') + '/' + filename, 'w') as fo:
        alignments_out, highest_score, len_x, len_y = main(args.corpus_folder_source.rstrip('/') + '/' + filename, args.corpus_folder_target.rstrip('/') + '/' + filename, longlength, scorecutoff)
        fo.write(alignments_out)
    bname = basename(filename)
    executionTime = (time.time() - startTime)
    print('Filename: ' + bname + ' of length (' + str(len_x) + ', ' + str(len_y) + '). Highest "score": ' + str(highest_score) + '. Execution time in seconds: ' + str(executionTime))


if __name__ == '__main__':
    filelist = [f for f in listdir(args.corpus_folder_source) if isfile(join(args.corpus_folder_target, f))]
    if args.num_proc == 0:
        run_proc = mp.cpu_count() - 1
    else:
        run_proc = int(args.num_proc)

    with mp.Pool(processes=run_proc) as pool:
        pool.map(main_multi, filelist)


    # for f in filelist:
    #     f_starttime = time.time()
    #     try:
    #         with open(args.output_folder.rstrip('/') + '/' + f, 'w') as fo:
    #             fo.write(main(args.corpus_folder_source.rstrip('/') + '/' + f, args.corpus_folder_target.rstrip('/') + '/' + f, longlength, scorecutoff))
    #     except Exception as e:
    #         print(e)
    #         print(f + ' has issues, not dealing with that!')
    #     print(f + ' processed. File processing time in seconds: ' + str(time.time() - f_starttime))

