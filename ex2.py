import argparse
import math
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

OUT = [None] * 28  # Outputs array corresponds to Output1..Output28 (indices 0..27)
VOCAB_SIZE = 300000

def development_set_preprocessing(path: str) -> List[str]:
    """
    Read events from a corpus file at the following format:
      - 4 lines per article: header line, newline ('\n'), article text line, newline ('\n')
    We ignore header lines as well as newline lines and return a flat list of tokens (events) from article text lines only.
    """
    events: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i in range(2, len(lines), 4):
                events.extend(lines[i].strip().split())
    except FileNotFoundError:
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    return events

# ---------------------------
# Unigram probabilities
# ---------------------------
def mle_prob(word: str, counts: Counter, N: int) -> float:
    """Unigram MLE: p(x)=c(x)/N."""
    if N == 0: # empty training set
        return 0.0
    return counts.get(word, 0) / N


def lidstone_prob(word: str, counts: Counter, N: int, lam: float, V: int = VOCAB_SIZE) -> float:
    """
    Lidstone smoothing:
        p_lid(x) = (c(x)+lam) / (N + lam*V)
    For unseen x: c(x)=0 -> lam/(N+lam*V)
    """
    denom = N + lam * V
    if denom == 0:
        return 0.0
    return (counts.get(word, 0) + lam) / denom


def heldout_parameters(train_counts: Counter, heldout_counts: Counter, V: int = VOCAB_SIZE) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Compute N_r and t_r for held-out estimation:
      N_r = number of types with c_T(x)=r (unique tokens)
      t_r = sum_{x: c_T(x)=r} c_H(x)
    """
    n_r: Dict[int, int] = defaultdict(int)
    t_r: Dict[int, int] = defaultdict(int)

    # r=0: types not seen in training
    n_r[0] = V - len(train_counts)
    t0 = 0
    for w, ch in heldout_counts.items():
        if w not in train_counts:
            t0 += ch
    t_r[0] = t0

    # r>0: types seen in training
    for w, r in train_counts.items():
        n_r[r] += 1
        t_r[r] += heldout_counts.get(w, 0)

    return n_r, t_r


def heldout_prob(word: str, train_counts: Counter, n_r: Dict[int, int], t_r: Dict[int, int], H_size: int) -> float:
    """
    Held-out unigram probability:
        p_H(x) = t_r[r] / (N_r[r] * |H|), where r=c_T(x).
    """
    r = train_counts.get(word, 0)
    return t_r.get(r, 0) / (n_r.get(r, 0) * H_size)


# ---------------------------
# Perplexity
# ---------------------------
def perplexity_unigram(events: List[str], prob_fn) -> float:
    """
    Perplexity = exp( - (1/n) * sum_i log p(w_i) )
    Uses natural logs; equivalent to any other log base.
    If any p(w_i)==0 => perplexity = inf.
    """
    n = len(events) # Validation set size
    if n == 0:
        return float("inf")
    log_sum = 0.0
    for w in events:
        p = prob_fn(w)
        if p <= 0.0:
            return float("inf")
        log_sum += math.log(p)
    return math.exp(-log_sum / n)


# ---------------------------
# Splits and search
# ---------------------------
def split_90_10(events: List[str]) -> Tuple[List[str], List[str]]:
    """Train = first round(0.9*|S|) events, Validation = rest."""
    cut = int(round(0.9 * len(events)))
    return events[:cut], events[cut:]


def grid_search_lambda(train_counts: Counter, train_size: int, validation: List[str], V: int = VOCAB_SIZE) -> Tuple[float, float]:
    """
    Search lambda in [0, 2] with step 0.01, pick lambda with minimal validation perplexity.
    Returns: (best_lambda, best_perplexity)
    """
    best_lam = 0.0
    best_pp = float("inf")
    for i in range(0, 201):  # 0.00..2.00
        lam = i / 100.0
        pp = perplexity_unigram(validation, lambda w: lidstone_prob(w, train_counts, train_size, lam, V))
        if pp < best_pp:
            best_pp = pp
            best_lam = lam
    return best_lam, best_pp


# ---------------------------
# Debug mass check
# ---------------------------
def debug(p_unseen: float, n0: int, seen_probs: Iterable[float]) -> float:
    """Return total probability mass: p_unseen*n0 + sum_{seen} p(x)."""
    sum_seen = sum(seen_probs)
    total_prob = (p_unseen * n0) + sum_seen
    return total_prob

# ---------------------------
# Output29 table
# ---------------------------
def build_table_29(
    train_size: int,
    N_ST: int,
    N_SH: int,
    n_r: Dict[int, int],
    t_r: Dict[int, int],
    best_lambda: float,
    V: int = VOCAB_SIZE,
) -> List[List[float]]:
    """
    Build Output29 rows for r=0..9:
      r, fMLE, fλ, fH, NTr, tr

    We follow the exercise description:
      - r is the frequency in S^T (the held-out training half)
      - fMLE: expected frequency under MLE on the same corpus -> equals r
      - fλ: expected frequency under Lidstone(best λ) for a word with count r in the same corpus
      - fH: expected frequency under held-out for a word with count r (depends only on r)
      - NTr: N_r from held-out (computed on S^T)
      - tr:  t_r from held-out
    """
    rows: List[List[float]] = []

    for r in range(0, 10):
        f_mle = float(r)

        # Lidstone expected frequency for any word with count r in the training corpus
        p_lid = (r + best_lambda) / (train_size + best_lambda * V)
        f_lid = p_lid * train_size

        # Held-out expected frequency depends only on r via t_r and N_r
        ntr = n_r.get(r, 0)
        tr = t_r.get(r, 0)
        p_ho = tr / (ntr * N_SH) 
        f_ho = p_ho * N_ST

        rows.append([f_mle, f_lid, f_ho, ntr, tr])

    return rows


# ---------------------------
# Writing output
# ---------------------------
def write_output(path: str, outputs: List, table: List[List[float]]) -> None:
    """
    Output file format (tab-delimited):
      #Students <student1 name> <student1 id> <student2 name> <student2 id>
      #Output1 <value>
      ...
      #Output28 <value>
      #Output29
      <10 lines of table: r=0..9, each row tab-delimited, rounded to 5 decimals>
    """
    with open(path, "w", encoding="utf-8") as f:
        # TODO: replace with your real details
        f.write("#Students\tStudent_Name1\tID1\tStudent_Name2\tID2\n")

        for i in range(28):
            f.write(f"#Output{i+1}\t{outputs[i]}\n")

        f.write("#Output29\n")
        for row in table:
            f.write(
                "\t".join(
                    f"{x:.5f}" if i in (1, 2) else str(int(x))
                    for i, x in enumerate(row)
                ) + "\n"
)


# ---------------------------
# Main
# ---------------------------
def main() -> None:

    parser = argparse.ArgumentParser(description="Smoothing methods for Unigram model")
    parser.add_argument("dev_set", type=str, help="Path to development set file")
    parser.add_argument("test_set", type=str, help="Path to test set file")
    parser.add_argument("input_word", type=str, help="Word to estimate probability for")
    parser.add_argument("output_file", type=str, help="Path to output file")
    args = parser.parse_args()

    OUT[0] = args.dev_set
    OUT[1] = args.test_set
    OUT[2] = args.input_word
    OUT[3] = args.output_file
    OUT[4] = VOCAB_SIZE
    OUT[5] = 1.0 / VOCAB_SIZE  # uniform

    # Read development events, ignoring headers
    dev_events = development_set_preprocessing(args.dev_set)
    OUT[6] = len(dev_events)  # Output7

    # Lidstone: 90/10 split, lambda selection
    train, validation = split_90_10(dev_events)
    OUT[7] = len(validation)  # Output8
    OUT[8] = len(train)  # Output9

    counts = Counter(train)               # Counts how many times each token appears
    train_size = len(train)               # Total number of tokens in the training set
    OUT[9] = len(counts)                     # Output10: observed vocab size
    OUT[10] = counts.get(args.input_word, 0) # Output11: count of input_word in the training set

    # Output12/13: MLE probabilities
    OUT[11] = mle_prob(args.input_word, counts, train_size)
    OUT[12] = mle_prob("unseen-word", counts, train_size)

    # Output14/15: Lidstone probabilities with lambda=0.10
    OUT[13] = lidstone_prob(args.input_word, counts, train_size, 0.10, VOCAB_SIZE)
    OUT[14] = lidstone_prob("unseen-word", counts, train_size, 0.10, VOCAB_SIZE)

    # Output16/17/18: validation perplexities for lambda in {0.01,0.10,1.00}
    OUT[15] = perplexity_unigram(validation, lambda w: lidstone_prob(w, counts, train_size, 0.01, VOCAB_SIZE))
    OUT[16] = perplexity_unigram(validation, lambda w: lidstone_prob(w, counts, train_size, 0.10, VOCAB_SIZE))
    OUT[17] = perplexity_unigram(validation, lambda w: lidstone_prob(w, counts, train_size, 1.00, VOCAB_SIZE))

    # Output19/20: best lambda and its minimized perplexity on validation
    best_lam, best_pp = grid_search_lambda(counts, train_size, validation, VOCAB_SIZE)
    OUT[18] = best_lam
    OUT[19] = best_pp

    # ------------------
    # Held-out: split dev into halves
    # ------------------
    half = len(dev_events) // 2
    S_T = dev_events[:half]
    S_H = dev_events[half:]
    OUT[20] = len(S_T)  # Output21
    OUT[21] = len(S_H)  # Output22

    counts_ST = Counter(S_T)
    counts_SH = Counter(S_H)
    n_r, t_r = heldout_parameters(counts_ST, counts_SH, VOCAB_SIZE)

    OUT[22] = heldout_prob(args.input_word, counts_ST, n_r, t_r, len(S_H))   # Output23
    OUT[23] = heldout_prob("unseen-word", counts_ST, n_r, t_r, len(S_H))     # Output24

    # ------------------
    # Debug: probability mass sums to 1 (prints to stderr; doesn't affect output format)
    # ------------------
    # Lidstone (best lambda) mass over the 90% training corpus:
    n0_lid = VOCAB_SIZE - len(counts)
    p_unseen_lid = lidstone_prob("unseen-word", counts, train_size, best_lam, VOCAB_SIZE)
    seen_probs_lid = (lidstone_prob(w, counts, train_size, best_lam, VOCAB_SIZE) for w in counts.keys())
    print(f"[debug] Lidstone mass total = {debug(p_unseen_lid, n0_lid, seen_probs_lid):.12f}", file=sys.stderr)

    # Held-out mass over S^T:
    n0_ho = VOCAB_SIZE - len(counts_ST)
    p_unseen_ho = heldout_prob("unseen-word", counts_ST, n_r, t_r, len(S_H))
    seen_probs_ho = (heldout_prob(w, counts_ST, n_r, t_r, len(S_H)) for w in counts_ST.keys())
    print(f"[debug] Held-out mass total = {debug(p_unseen_ho, n0_ho, seen_probs_ho):.12f}", file=sys.stderr)

    # ------------------
    # Evaluate on test set
    # ------------------
    test_events = development_set_preprocessing(args.test_set)
    OUT[24] = len(test_events)  # Output25

    # Lidstone on test: train on FULL dev set with the selected lambda
    counts_dev_full = Counter(dev_events)
    N_dev_full = len(dev_events)
    OUT[25] = perplexity_unigram(test_events, lambda w: lidstone_prob(w, counts, train_size, best_lam, VOCAB_SIZE))  # Output26

    # Held-out perplexity on test: use held-out probabilities learned from S^T/S^H
    OUT[26] = perplexity_unigram(test_events, lambda w: heldout_prob(w, counts_ST, n_r, t_r, len(S_H)))  # Output27

    OUT[27] = "L" if OUT[25] < OUT[26] else "H"  # Output28

    # ------------------
    # Output29 table (r=0..9) from the held-out estimation (S^T/S^H) + best lambda
    # ------------------
    table = build_table_29(train_size, len(S_T),len(S_H), n_r, t_r, best_lam, VOCAB_SIZE)

    # Write output file
    write_output(args.output_file, OUT, table)

if __name__ == "__main__":
    main()
