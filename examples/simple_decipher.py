import random
from collections import Counter
import numpy as np
from tqdm import tqdm
from hmmlearn import hmm

from nltk.corpus import brown
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, KneserNeyInterpolated, Laplace
from nltk.util import pad_sequence

from hmm_pytorch import HMM
import torch
import itertools


def score_deciphered_ciphertext(decipherment: str, plaintext: str):
    """Score a decipherment by comparing it to the plaintext.

    Args:
        decipherment (str): The decipherment to score.
        plaintext (str): The plaintext.

    Returns:
        float: The score of the decipherment.
    """
    score = 0
    for c_pt, c_ct in zip(plaintext, decipherment):
        if c_pt == c_ct:
            score += 1
    accuracy = score / len(plaintext)
    return accuracy


def project_emission_down(emission, vocab_pt, vocab_ct):
    emission = np.reshape(emission, (len(vocab_pt), len(vocab_pt), -1))
    emission = emission[:, 0, :]
    assert emission.shape == (len(vocab_pt), len(vocab_ct))
    return emission


def convert_emission_to_key(
    emission: np.ndarray,
    int2pt: dict,
    int2ct: dict
) -> dict:
    """Convert emission matrix to ct2pt key."""
    ct2pt = {}
    for i_ct, c_ct in int2ct.items():
        i_pt = np.argmax(emission[:, i_ct])
        c_pt = int2pt[i_pt]
        ct2pt[c_ct] = c_pt
    return ct2pt


def convert_key_to_emission(
    ct2pt: dict,
    pt2int: dict,
    ct2int: dict
) -> np.ndarray:
    """Convert ct2pt key to emission matrix."""
    emission = np.zeros((len(pt2int), len(ct2int)))
    for c_ct, i_ct in ct2int.items():
        i_pt = pt2int[ct2pt[c_ct]]
        emission[i_pt, i_ct] = 0.5
    
    # distribute remaining probability mass
    for i, row in enumerate(emission):
        if (np.sum(row)) < 1:
            emission[i, :] += (1 - np.sum(row)) / len(row)
    
    # if row sums to zero, set all values to 1 / len(cols)
    # for i, row in enumerate(emission):
    #     if (np.sum(row)) == 0:            
    #         emission[i, :] = 1 / len(row)

    return emission


def score_predicted_key(pred_ct2pt: dict, gold_ct2pt: dict):
    """Score a ct2pt key by comparing it to the gold ct2pt key."""
    score = 0
    for c_ct, c_pt in pred_ct2pt.items():
        if c_pt == gold_ct2pt[c_ct]:
            score += 1
    accuracy = score / len(pred_ct2pt)
    return accuracy


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_process_corpus(n_sentences=1000):
    sents = brown.sents()[:n_sentences]
    # lowercase and remove special characters
    sents = [[word.lower() for word in sent if word.isalpha()]
             for sent in sents]
    # join words into sentences
    sents = [' '.join(sent) for sent in sents]
    # tokenize the sentences into characters
    chars = [list(sent) for sent in sents]
    return chars


def train_ngram_lm(text, order=2):
    train_data, padded_sents = padded_everygram_pipeline(
        order=order, text=text)
    lm = MLE(order=order)
    # lm = Laplace(order=order)
    print(f"Training {order}-gram language model...")
    lm.fit(train_data, padded_sents)
    print("Done.")
    return lm


def get_plaintext_vocab(lm):
    vocab_pt = sorted(lm.vocab)
    # drop <UNK>
    vocab_pt.remove('<UNK>')

    pt2int = {c: i for i, c in enumerate(vocab_pt)}
    int2pt = {i: c for i, c in enumerate(vocab_pt)}

    return vocab_pt, pt2int, int2pt


def get_ciphertext_vocab(ciphertext, order=2):
    # pad ciphertext
    # ciphertext = list(pad_sequence(ciphertext,
    # n=order, pad_left=True,
    # left_pad_symbol="<s>",
    # pad_right=True,
    # right_pad_symbol="</s>",))
    _, ciphertext = padded_everygram_pipeline(
        order=order, text=[list(ciphertext)])
    ciphertext = list(ciphertext)
    # print(f"{list(ciphertext)=}")
    # foo
    vocab_ct = sorted(set(ciphertext))
    ct2int = {c: i for i, c in enumerate(vocab_ct)}
    int2ct = {i: c for i, c in enumerate(vocab_ct)}
    return vocab_ct, ct2int, int2ct, ciphertext


def get_simple_sub_key(vocab_pt):
    # create a mapping from plaintext to ciphertext
    # but make sure <s> maps to <s> and </s> maps to </s> and <UNK> maps to <UNK> and ' ' maps to ' '
    mapping = {'<s>': '<s>', '</s>': '</s>', ' ': ' '}
    # add the remaining characters
    for char in vocab_pt:
        if char not in mapping:
            mapping[char] = random.choice(
                [c for c in vocab_pt if c not in mapping.values()])

    pt2ct = mapping
    ct2pt = {v: k for k, v in mapping.items()}
    return pt2ct, ct2pt


def encipher_text(text, pt2ct):
    ciphertext = [pt2ct[c] for c in text]
    return ciphertext


def convert_2gram_lm_to_tm(lm, vocab):
    tm = np.zeros((len(vocab), len(vocab)))
    for i, c in enumerate(vocab):
        for j, d in enumerate(vocab):
            if c == '</s>' and d == '</s>':
                tm[i, j] = 1
            else:
                tm[i, j] = lm.score(d, [c])
    return tm


def convert_3gram_lm_to_tm(lm, unigrams):

    bigrams = []
    for c_1 in unigrams:
        for c_2 in unigrams:
            # we set the second element of the tuple
            # to c_1 to ensure the bigrams that emit
            # the same character are grouped together
            bigrams.append((c_2, c_1))

    bigram2idx = {c: i for i, c in enumerate(bigrams)}
    idx2bigram = {i: c for i, c in enumerate(bigrams)}

    tm = np.zeros((len(bigrams), len(bigrams)))

    for i, (a, b) in enumerate(bigrams):
        for j, (c, d) in enumerate(bigrams):
            if b == c:
                tm[i, j] = lm.score(d, [a, b])

    # for any rows that sum to 0, find the valid columns that follow
    # a row (first char of column == second char of row). Set equal
    # probability to all valid columns.
    for i, row in enumerate(tm):
        if (np.sum(row)) == 0:
            cols = [j for j, (c, _) in enumerate(
                bigrams) if c == idx2bigram[i][1]]
            tm[i, cols] = 1 / len(cols)

    assert np.allclose(np.sum(tm, axis=1), 1)

    return tm, idx2bigram, bigram2idx


def decipher(
    random_restarts: int = 10,
    training_sequences: int = 1_00_000,
    ngrams=2,    
    hmm_impl='hmmlearn', # ['hmmlearn', 'pytorch']
    seed=42,
    initialize_with_gold=False,
) -> None:

    set_seed(seed=seed)
    text = load_and_process_corpus(n_sentences=training_sequences)
    lm = train_ngram_lm(text, order=ngrams)

    vocab_pt, pt2int, int2pt = get_plaintext_vocab(lm)
    print(f"Plaintext vocab size = {len(vocab_pt)}")
    pt2ct, ct2pt = get_simple_sub_key(vocab_pt)
    text_to_encipher = ''.join(text[5])
    ciphertext = encipher_text(text_to_encipher, pt2ct)
    
    # set order=2 because only the hidden states conditioned on the previous two hidden states
    # the emissions are only conditioned on its parent hidden state
    vocab_ct, ct2int, int2ct, ciphertext = get_ciphertext_vocab(
        ciphertext, order=2)
    print(f"Ciphertxt vocab size = {len(vocab_ct)}")
    print(f"Plaintext: {''.join([ct2pt[c] for c in ciphertext])}")
    print(f"Ciphertxt: {''.join(ciphertext)}")
    print(f"Cipher length: {len(ciphertext)}")

    # Observations
    X_train = np.array([[ct2int[c] for c in ciphertext]]).T
    
    if ngrams == 2:
        transmat = convert_2gram_lm_to_tm(lm, vocab_pt)
        startprob = np.zeros(len(vocab_pt))
        startprob += 1e-15
        startprob[pt2int['<s>']] = 1
        startprob /= np.sum(startprob)        
        n_states = len(vocab_pt)        
        if hmm_impl == 'pytorch':
            transmat = torch.tensor(transmat)
            startprob = torch.tensor(startprob)
            X_train = torch.tensor(X_train)
    elif ngrams == 3:
        transmat, idx2bigram, bigram2idx = convert_3gram_lm_to_tm(lm, vocab_pt)
        startprob = np.zeros(len(bigram2idx))
        startprob[bigram2idx[('<s>', '<s>')]] = 1
        n_states = len(bigram2idx)    
    else:
        raise ValueError(f"{ngrams=}")

    # Train HMM    
    best_score = best_model = best_idx = None

    f = tqdm(range(random_restarts))
    for idx in f:
        f.set_description(f"Fitting {idx}")

        if hmm_impl == 'hmmlearn':
            model = hmm.CategoricalHMM(
                n_components=n_states,
                random_state=idx,
                params='e',
                init_params='e' if ngrams == 2 and not initialize_with_gold else '',
                implementation='scaling',  # faster than 'log'
                # implementation='log',
                order=ngrams-1,
            )
        elif hmm_impl == 'pytorch':
            model = HMM(
                n_components=n_states,
                n_features=len(vocab_ct),
                random_state=idx,
                params='e',
                init_params='e',
                order=1,
            )
        else:
            raise ValueError(f"{hmm_impl=}")

        model.startprob_ = startprob
        model.transmat_ = transmat
        if initialize_with_gold and ngrams == 2:
            # print("Initializing with gold")
            emissionprob = convert_key_to_emission(ct2pt, pt2int, ct2int)
            if hmm_impl == 'pytorch':
                emissionprob = torch.tensor(emissionprob)
            model.emissionprob_ = emissionprob

        # initialize emission matrix
        if ngrams == 3:
            np.random.seed(idx)
            emissionmat = np.random.rand(
                len(vocab_pt), len(vocab_ct))  # [pt, ct]
            emissionmat[pt2int['<s>'], ct2int['<s>']] = 1e6
            emissionmat[pt2int['</s>'], ct2int['</s>']] = 1e6
            # emissionmat[[pt2int['</s>'] + len(vocab_pt)*i for i in range(len(vocab_pt))], ct2int['</s>']] = 1e6
            emissionmat = emissionmat / \
                np.sum(emissionmat, axis=1, keepdims=True)

            # need to project such that emission matrix rows == transition matrix rows
            # rows (bigrams) with the same 2nd character share the same emission probabilities
            emissionmat = np.repeat(emissionmat, len(
                vocab_pt), axis=0)  # [pt^2, ct]
            model.emissionprob_ = emissionmat

        # print(f"{model.emissionprob_[:len(vocab_pt)+2, :]=}")
        # print(f"{model.emissionprob_[len(vocab_pt):2*len(vocab_pt), :]=}")
        # foo

        model.fit(X_train)
        score = model.score(X_train)

        # print(f'Model #{idx}\tScore: {score}')
        # print(f"{model.emissionprob_[:len(vocab_pt), :]=}")
        # print(f"{model.emissionprob_[len(vocab_pt):2*len(vocab_pt), :]=}")
        # foo

        if best_score is None or score > best_score:
            best_model = model
            best_score = score
            best_idx = idx
    
    print(f"best model: {best_idx}")
    print(f"best score: {best_score}")

    ### Evaluation ###
    # best_model.algorithm = 'viterbi'
    # states = best_model.predict(X_train)

    if ngrams == 2:
        # pred_plaintext = ''.join([int2pt[s] for s in states])
        # enciphered_text = ''.join([ct2pt[c] for c in ciphertext])
        # ser = score_deciphered_ciphertext(pred_plaintext, enciphered_text)
        # print(f"predicted plaintext (Viterbi): {pred_plaintext}")
        # print(f"symbol error rate: {round(ser, 4)}")

        best_model.algorithm = 'map'
        states = best_model.predict(X_train)
        pred_plaintext = ''.join([int2pt[s] for s in states])
        enciphered_text = ''.join([ct2pt[c] for c in ciphertext])
        ser = score_deciphered_ciphertext(pred_plaintext, enciphered_text)
        print(f"predicted plaintext (MAP): {pred_plaintext}")
        print(f"symbol error rate: {round(ser, 4)}")


    elif ngrams == 3:        
        pred_plaintext = ''.join([idx2bigram[s][-1] for s in states])
        enciphered_text = ''.join([ct2pt[c] for c in ciphertext])
        ser = score_deciphered_ciphertext(pred_plaintext, enciphered_text)
        print(f"predicted plaintext (Viterbi): {pred_plaintext}")
        print(f"symbol error rate: {round(ser, 4)}")

        emissionmat = project_emission_down(
            best_model.emissionprob_, vocab_pt, vocab_ct)
        pred_ct2pt = convert_emission_to_key(emissionmat, int2pt, int2ct)
        mer = score_predicted_key(pred_ct2pt, ct2pt)
        # print(f"{pred_ct2pt=}")
        print(f"mapping error rate (Key): {mer=}")

        pred_plaintext_key = ''.join([pred_ct2pt[c] for c in ciphertext])
        ser_key = score_deciphered_ciphertext(
            pred_plaintext_key, enciphered_text)
        print(f"predicted plaintext (Key): {pred_plaintext_key}")
        print(f"symbol error rate: {round(ser_key, 4)}")

        best_model.algorithm = 'map'
        states = best_model.predict(X_train)
        pred_plaintext_mbr = ''.join([idx2bigram[s][-1] for s in states])    
        ser_mbr = score_deciphered_ciphertext(pred_plaintext_mbr, enciphered_text)
        print(f"predicted plaintext (MBR): {pred_plaintext_mbr}")
        print(f"symbol error rate: {round(ser_mbr, 4)}")
    


def main(
    text_to_encipher: str = 'legislators',
    seed: int = 42,
):
    decipher(ngrams=2, seed=seed, hmm_impl='hmmlearn', random_restarts=1, initialize_with_gold=True)
    decipher(ngrams=2, seed=seed, hmm_impl='pytorch', random_restarts=1, initialize_with_gold=True)
        


if __name__ == '__main__':
    main()
