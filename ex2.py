import argparse
from collections import Counter
import sys

VOCAB_SIZE = 300000

OUTPUT = [None] * 40

def init():
    # process input: the program gets accept the following 4 arguments in this exact order:
    # < development set filename > < test set filename > < INPUT WORD > < output filename >
    # then write to the first sixth of Output array the following values: 
    # development set file name,  test set file name, INPUT WORD,  output file name, the language vocabulary size (300,000) and P_uniform.
    parser = argparse.ArgumentParser(description='Language Model Estimation')
    parser.add_argument('dev_set', type=str, help='Path to the development set file')
    parser.add_argument('test_set', type=str, help='Path to the test set file')
    parser.add_argument('input_word', type=str, help='The word to estimate probability for')
    parser.add_argument('output_file', type=str, help='Path to the output file')  
    args = parser.parse_args()
    # P_uniform = 1 / V
    p_uniform = 1.0 / VOCAB_SIZE
    # save values to array
    OUTPUT[0] = args.dev_set
    OUTPUT[1] = args.test_set
    OUTPUT[2] = args.input_word
    OUTPUT[3] = args.output_file
    OUTPUT[4] = VOCAB_SIZE
    OUTPUT[5] = p_uniform
    return args
    # < development set filename > < test set filename > are .txt files. 
    # There are 2 lines for each article in the input files. The first one is the article header and the
    # second is the article itself. When developing and testing your model you should only consider
    # the article itself (and NOT its header line). 


def development_set_preprocessing(dev_set_path):
    # given the deveploment set path, compute the total number of events in the development set |S| (include repetition)
    # write to the 7_th item of Output this value
    # assume the words at the corpus are tokenized (seperated by space)
    # Consider the text as a sequence of events (words) that are separated by white spaces (usually a single space or a
    # new line). Basically everything between 2 white spaces is an event.
    all_events = []
    try:
        with open(dev_set_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # iterate every 2 lines to skip headers
            for i in range(2, len(lines), 4):
                words = lines[i].strip().split()
                all_events.extend(words)
    except FileNotFoundError:
        print(f"Error: File {dev_set_path} not found.")
        sys.exit(1)
    # save value to output
    OUTPUT[6] = len(all_events)
    print(len(all_events))
    # return list of all events    
    return all_events
 

def Lidstone_model_training ():
    pass
    # Step 1: (internal function)
    # split the development set into a training set with exactly the first
    # 90% of the words in S (should contain the first (round(0.9 ∗|S|)) words and a validation
    # set with the rest 10% of the words.

    # write to 8_th and 9_th cells the number of events in the validation set and training set, appropriately. 
    # write to 10_th cell the number of different events at training set
    # write to the 11_th cell the number of times the event INPUT WORD appears in the training set
    # write to the 12_th cell the Maximum Likelihood Estimate of INPUT WORD using unigram_prob
    # write to the 13_cell the Maximum Likelihood Estimate of 'unseen-word' using unigram_prob
    # write to the 14th cell the P(Event = INPUT WORD) as estimated by your model using λ = 0.10 (call unigram_prob_lidstone)
    # write to the 15th cell the P(Event = 'unseen-word') as estimated by your model using λ = 0.10 (call unigram_prob_lidstone)
    # write to the 16th cell the perplexity on the validation set using λ = 0.01 (call calculate_perplexity)
    # write to the 17th cell the perplexity on the validation set using λ = 0.10 (call calculate_perplexity)
    # write to the 18th cell the perplexity on the validation set using λ = 1.00 (call calculate_perplexity)
    # cell call to grid search (assign to the 19th cell the optimum λ).
    # write the the 20th the value return from calculate_perplexity with the optimum lambda. 

def held_out_model_training(all_events, input_word):
    # Split the development set into exactly 2 halfes.
    half = len(all_events) // 2
    training_set = all_events[:half]
    held_out_set = all_events[half:]
    # write to the 21 cell the number of events in the first halve (include repetition). mark: S^T, training set
    # write to the 22 cell the number of events in the second halve (include repetition). mark: S^H, held-out set
    OUTPUT[20] = len(training_set)
    OUTPUT[21] = len(held_out_set)
    # call calculate_held_out_parameters.
    counts_train = Counter(training_set)  # calculate counts of events in S^T
    counts_held_out = Counter(held_out_set)  # calculate counts of events in S^H
    n_r, t_r = calculate_held_out_parameters(counts_train, counts_held_out)
    # write to the 23th P(Event = INPUT WORD) as estimated by the unigram with held out smoothing model.
    prob_input = unigram_prob_held_out(input_word, counts_train, n_r, t_r, len(held_out_set))
    OUTPUT[22] = prob_input
    print(f"{input_word} probabiliry: {prob_input}" )
    # write to the 24th P(Event = 'unseen word') as estimated by unigram with held out smoothing model.
    prob_unseen = unigram_prob_held_out('unseen-word', counts_train, n_r, t_r, len(held_out_set))
    OUTPUT[23] = prob_unseen
    print(f"unseen-word probabiliry: {prob_unseen}" )
    return counts_train, n_r, t_r, len(held_out_set)

def calculate_held_out_parameters(counts_train, counts_held_out):
    # The equation: 
    # $p_{H_0}(x : c_T(x) = r) = \frac{t_r / N_r}{|H|} = \frac{\sum_{x : c_T(x) = r} c_H(x)}{N_r |H|}$
    n_r = Counter()
    t_r = Counter()
    # define n_0
    n_r[0] = VOCAB_SIZE - len(counts_train)
    t_0 = 0
    # calculate t_0
    for word, count in counts_held_out.items():
        if word not in counts_train:
            t_0 += count
    t_r[0] = t_0
    # calculate t_r
    for word, r in counts_train.items():
        n_r[r] += 1
        t_r[r] += counts_held_out[word]
    return n_r, t_r

def calculate_perplexity():
    pass
    # given a specific lambda, calculate the model perplexity on the validation set.

def grid_search():
    pass
    # search for lambda value minimize the perplexity on the validation set
    # lambda is at the range [0, 2]. λ values should be specified up to two digits after the decimal point 
    # at loop, call to calculate_perplexity with the current λ
    # write to the 19th cell the λ that minimizes the perplexity

def unigram_prob():
    pass
    # according to MLE based on the training set for word x
    # equation: f(x) / N. 
    # where N is the Output 9th cell. 

def unigram_prob_lidstone():
    pass
    # given a specific lambda and the training set, calculate the prob of word x 

def unigram_prob_held_out(word, counts_train, n_r, t_r, size_h):
    r = counts_train[word]  # find r of 'word'
    return t_r[r] / (n_r[r] * size_h)  # return p (x=word)

def debug():
    pass
    # make sure the probabilities are summed up to 1 at the lidstone and held out models. 
    # equation: $p(x^{*})\, n_{0} + \sum_{x : \operatorname{count}(x) > 0} p(x) = 1$
    # where p(x∗) is the probability of any unseen event and n0 is the number of such events.
    # where count(x) is the amount of apperances at the relevent training set 

def evaluate(): pass
    # write to the 25th cell the total number of events in the test set (present at 2th cell).
    # writh to the 26th cell the perplexity of the lidstone model with optimum lambda on the test set
    # writh to the 27th cell the perplexity of the held out model 
    # write to the 28th cell 'L' whether the perplexity if lidstone is lower than held out. otherwise, write 'H'. 
    # write to the 29cell the following table: 
    # Intructions:
        # Numbers in each row should be tab delimited (i.e. separated by a tab character). Round your results in this table to
        # 5 digits after the decimal point.
        # Don't write the header column. 
        # first column: r values denote event frequencies in the training corpus (r \in [0, 9])
        # second column: for each x \in test set, estimate the frequency according p(x) * |training_set|
        # calculate expected frequency over all the x \in test set. 
        # third column: same for the the held out estimation. 
        # forth column: number of events of frequency r in the S^T of the development set.
        # fifth column: for each x\in S^T with frequency r, calculate the frequency of x at S^H. sum over all x. 


def write_to_file(): pass
# Your output file should include exactly the following lines, in the following order. Each
#  line should be tab delimited (i.e. a tab character between every two strings).
#Students <student1 name> <student1 id> <student2 name> <student2 id>
#Output1 < value >
#  ...
#Output28 < value >
#Output29
#<10 lines of the table described in Output29>
# Output number values containing an exponent symbol (‘e’) may appear in either uppercase
# or lowercase. E.g. both 1.1111111e6 and 1.1111111E6 are acceptable.
# print the values based on OUTPUT array. 
# print to outputfile (present at 4th cell).
    


if __name__ == "__main__":
    args = init()
    print(args)
    dev_set = development_set_preprocessing(args.dev_set)
    counts_train, n_r, t_r, ho_set_len = held_out_model_training(dev_set, args.input_word)


