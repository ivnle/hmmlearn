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

    # get product of vocab x vocab
    bigrams = list(itertools.product(unigrams, unigrams))
    idx2bigram = {}
    tm = np.zeros((len(bigrams), len(bigrams)))    
    
    for i, (a, b) in enumerate(bigrams):
        idx2bigram[i] = (a, b)
        for j, (c, d) in enumerate(bigrams):
            if b == c:
                tm[i, j] = lm.score(d, [a, b])

    # for all rows that sum to 0, set columns to uniform distribution
    for i, row in enumerate(tm):
        if np.sum(row) == 0:
            tm[i, :] = 1/len(bigrams)
    
    # Check that rows sum to 1
    # print(tm)

    # print(tm[chars2index[('<s>', '<s>')], :])
    # print(np.sum(tm, axis=1))
    assert np.allclose(np.sum(tm, axis=1), 1)
    # project tm from (n**2, n) to (n**2, n**2) such that
    # each column of tm is duplicated n times but are contiguous
    # tm = np.repeat(tm, len(bigrams), axis=1)  # (shape^2, shape^2)
    # TODO this is wrong, use repeat for emission matrix
    # consider looping over two arrays, each array is comprised of (char_1, char_2)
    # use collections product
    
    # print(tm)
    # print(tm.shape)
    # foo

    return tm, idx2bigram


def decipher(ngrams=1 seed=42):
    
    set_seed(seed=seed)
    text = load_and_process_corpus(n_sentences=10000)    
    lm = train_ngram_lm(text, order=ngrams)
    vocab_pt, pt2int, int2pt = get_plaintext_vocab(lm)
    pt2ct, ct2pt = get_simple_sub_key(vocab_pt)
    text_to_encipher = ''.join(text[5])
    ciphertext = encipher_text(text_to_encipher, pt2ct)
    vocab_ct, ct2int, int2ct, ciphertext = get_ciphertext_vocab(ciphertext)    
    
    if ngrams == 1:
        transmat = convert_2gram_lm_to_tm(lm, vocab_pt)
        startprob = np.zeros(len(vocab_pt))
        startprob[pt2int['<s>']] = 1
        X_train = np.array([[ct2int[c] for c in ciphertext]]).T
    elif ngrams == 2:
        transmat, idx2bigram = convert_3gram_lm_to_tm(lm, vocab_pt)
        bigram2idx = {v: k for k, v in idx2bigram.items()}
        
    else:
        raise ValueError(f"{ngrams=}")

    # Train HMM
    best_score = best_model = best_idx = None
    n_fits = 1

def main(
    text_to_encipher: str = 'legislators',
    seed: int = 42,
):
    set_seed(seed=42)

    # text = load_and_process_corpus(n_sentences=1000000)
    text = load_and_process_corpus(n_sentences=10000)
    # print(text[5])
    # print(repr(''.join(text[5])))

    # train bigram language model on text
    lm = train_ngram_lm(text, order=3)
    # print(f"{lm.score('a', ['b'])=}")
    # print(f"{lm.counts['b']=}")
    # print(f"{lm.counts[['b']]['a']=}")
    # print(f"{lm.counts[['b']]['a']/lm.counts['b']=}")

    vocab_pt, pt2int, int2pt = get_plaintext_vocab(lm)
    print(f"{vocab_pt=}")
    # print(f"{pt2int=}")
    # print(f"{int2pt=}")
    pt2ct, ct2pt = get_simple_sub_key(vocab_pt)
    # print(f"{pt2ct=}")
    # print(f"{ct2pt=}")
    text_to_encipher = ''.join(text[5])
    print(f"{text_to_encipher=}")
    ciphertext = encipher_text(text_to_encipher, pt2ct)
    # sanity check that ciphertext decodes to plaintext
    # plaintext = ''.join([ct2pt[c] for c in ciphertext])
    # print(f"{plaintext=}")

    vocab_ct, ct2int, int2ct, ciphertext = get_ciphertext_vocab(ciphertext)
    # print(f"{vocab_ct=}")
    # print(f"{ct2int=}")
    # print(f"{int2ct=}")
    print(f"{ciphertext=}")

    # ct_ints = [ct2int[c] for c in ciphertext]
    # print(f"{ct_ints=}")

    # check how LM scores ciphertext
    # ngrams, _ = padded_everygram_pipeline(order=2, text=[list(text_to_encipher)])
    # convert generator to list
    # for n in ngrams:
    #     grams = [x for x in n]
    #     print(f"{grams=}")
    #     print(lm.entropy(grams))
    # print(f"{ngrams=}")
    # print(lm.entropy(ngrams))

    # convert LM to transition matrix
    # transmat = convert_2gram_lm_to_tm(lm, vocab_pt)
    transmat, idx2bigram = convert_3gram_lm_to_tm(lm, vocab_pt)
    bigram2idx = {v: k for k, v in idx2bigram.items()}
    print(f"{transmat.shape=}")
    # First hidden state is always <s>
    startprob = np.zeros(len(vocab_pt))
    startprob[pt2int['<s>']] = 1
    print(f"{startprob.shape=}")
    print(f"{startprob=}")

    X_train = np.array([[ct2int[c] for c in ciphertext]]).T
    # print(f"{X_train=}")
    print(f"{X_train.shape=}")

    # Train HMM
    best_score = best_model = best_idx = None
    n_fits = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(f"{device=}")
    X_train = torch.from_numpy(X_train).to(device)

    f = tqdm(range(n_fits))
    for idx in f:
        f.set_description(f"Fitting {idx}")
        model = HMM(
            n_states=len(vocab_pt),
            n_obs=len(vocab_ct),
            start_prob=torch.from_numpy(startprob),
            transition=torch.from_numpy(transmat),
            random_state=idx,
            device=device,
        )
        model = model.to(device)

        print(f"{model.transition.is_cuda=}")
        print(f"{model.emission.is_cuda=}")
        print(f"{model.start_prob.is_cuda=}")

        print(f"{model.start_prob=}")
        # update emission matrix until convergence
        init_score = model.score_obs(X_train)
        print(f"{init_score=}")
        # foo

        tol = 1e-6
        max_iter = 100000
        last_score = -np.inf
        for i in range(max_iter):
            model.update_emission(X_train)
            # check if score has converged
            score = model.score_obs(X_train)
            if abs(last_score - score) < tol:
                break
            last_score = score

        if best_score is None or score > best_score:
            best_model = model
            best_score = score
            best_idx = idx
        print(f"{best_score=}")
        print(f"{score=}")
    foo

    f = tqdm(range(n_fits))
    for idx in f:
        f.set_description(f"Fitting {idx}")
        model = hmm.CategoricalHMM(
            n_components=len(vocab_pt),
            random_state=idx,
            params='e',
            init_params='e'
        )

        # model.n_features = vocab_sz

        # set startprob to alawys start on first state
        # model.startprob_ = np.zeros(plaintext_vocab_sz)
        # model.startprob_[0] = 1
        # set starprob to uniform
        model.startprob_ = startprob

        # set transition probabilities from language model
        model.transmat_ = transmat

        # set emission probabilities to random
        # model.emissionprob_ = np.random.rand(vocab_sz, vocab_sz) # plaintext -> ciphertext
        # # normalize the emission probabilities
        # model.emissionprob_ /= model.emissionprob_.sum(axis=1)[:, np.newaxis]
        # print(f"{model.emissionprob_.shape=}")

        model.fit(X_train)
        score = model.score(X_train)
        # print(f'Model #{idx}\tScore: {score}')
        if best_score is None or score > best_score:
            best_model = model
            best_score = score
            best_idx = idx

    # use the Viterbi algorithm to predict the most likely sequence of states
    # given the model
    states = best_model.predict(X_train)
    print(f"{states=}")
    # convert states to plaintext
    pred_plaintext = ''.join([int2pt[s] for s in states])
    print(f"{pred_plaintext=}")
    print(f"{text_to_encipher=}")
    target_states = np.array([pt2int[ct2pt[c]] for c in ciphertext])
    print(f"{target_states=}")
    # print matches between predicted and target states
    matches = np.where(states == target_states)[0]
    print(f"{matches=}")
    print(f"{len(matches)/len(states)=}")

    # take argmax of emission probabilities to get most likely ciphertext -> plaintext mapping
    pred_key = np.argmax(best_model.emissionprob_, axis=0)
    print(f"{pred_key=}")
    # convert to plaintext
    # remove last dimension from X_train
    X_train = X_train.squeeze()
    pred_states = [pred_key[c_int] for c_int in X_train]
    print(f"{pred_states=}")
    pred_plaintext = ''.join([int2pt[s] for s in pred_states])
    print(f"{pred_plaintext=}")


if __name__ == '__main__':
    main()
