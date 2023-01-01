"""k
Simple Substitution Cipher Example
------------------------
In this example, we'll use a simple substitution cipher to demonstrate
how to learn the emission probabilities of a Hidden Markov Model (HMM).
The transition probabilities are fixed according to a bigram language
model. We observe some ciphertext and wish to learn the emission probabilities
that maximize the likelihood of the observed ciphertext. Once we have learned
the emission probabilities, we can use the Viterbi algorithm to find the most
likely plaintext.
"""

#%%
# First, lets create a simple substitution cipher.
import string
import random
from collections import Counter

# set seed
random.seed(43)

#%%
# let's create a bigram character level language model.
# We'll use the Brown corpus to train the language model.
from nltk.corpus import brown
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

# get the first 1000 sentences from the Brown corpus
brown_sents = brown.sents()[:1_000_000]
# lowercase and remove special characters
brown_sents = [[word.lower() for word in sent if word.isalpha()] for sent in brown_sents]
# join words into sentences
brown_sents = [' '.join(sent) for sent in brown_sents]
print(f"{brown_sents[0]=}")
#%%
# tokenize the sentences into characters
brown_sents = [list(sent) for sent in brown_sents]
print(f"{brown_sents[0]=}")

#%%
# get counts of all unique characters
char_counts = Counter()
for sent in brown_sents:
    char_counts.update(sent)    
print(f"{char_counts=}")

# sort the characters by count
char_counts = dict(sorted(char_counts.items(), key=lambda item: item[1], reverse=True))

#%%
# plot distribution of characters
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(char_counts.keys(), char_counts.values())
ax.set_title('Character Distribution')
ax.set_xlabel('Character')
ax.set_ylabel('Count')
plt.show()

#%%
# train a bigram language model
n = 2
train_data, padded_sents = padded_everygram_pipeline(order=n, text=brown_sents)
lm = MLE(order=n)
lm.fit(train_data, padded_sents)

#%%
lm.entropy([list('fr')])

#%%
plaintext_vocab_sz = len(lm.vocab)
print(f"{plaintext_vocab_sz=}")

#%%
#%% Convert LM to transition matrix
import numpy as np

# get the vocabulary
vocab = sorted(lm.vocab)
# drop <UNK>
vocab.remove('<UNK>')
print(f"{vocab=}")
# get the vocabulary size
vocab_sz = len(vocab)
# create a transition matrix
transmat = np.zeros((vocab_sz, vocab_sz))
# fill the transition matrix
for i, char in enumerate(vocab):
    for j, next_char in enumerate(vocab):
        if next_char == '</s>' and char == '</s>':
            transmat[i, j] = 1
        else:
            transmat[i, j] = lm.score(next_char, [char])

print(f"{transmat.shape=}")
# sum the rows to make sure they sum to 1
print(f"{transmat.sum(axis=1)=}")

#%% Create a simple substitution cipher
# create a mapping from plaintext to ciphertext
# but make sure <s> maps to <s> and </s> maps to </s> and <UNK> maps to <UNK> and ' ' maps to ' '
mapping = {'<s>': '<s>', '</s>': '</s>', '<UNK>': '<UNK>', ' ': ' '}
# add the remaining characters
for char in vocab:
    if char not in mapping:
        mapping[char] = random.choice([c for c in vocab if c not in mapping.values()])

pt2ct = mapping
ct2pt = {v: k for k, v in mapping.items()}
print(f"{pt2ct=}")
print(f"{ct2pt=}")

#%%
# set plaintext and ciphertext
plaintext = 'grand jury said friday an investigation'
ciphertext = ''.join([pt2ct[char] for char in plaintext])
print(f"{plaintext=}")
print(f"{ciphertext=}")

#%%
# Convert ciphertext and plaintext to integers
plaintext_ints = [vocab.index(char) for char in plaintext]
ciphertext_ints = [vocab.index(char) for char in ciphertext]

print(f"{plaintext_ints=}")
print(f"{ciphertext_ints=}")

#%%
# convert ciphertext_ints to plaintext
deciphered = [ct2pt[vocab[i]] for i in ciphertext_ints]
print(f"{deciphered=}")

#%%

from hmmlearn import hmm

X_train = np.array([ciphertext_ints]).T
print(f"{X_train=}")


#%%
best_score = best_model = None
n_fits = 10000
np.random.seed(13)
for idx in range(n_fits):
    model = hmm.CategoricalHMM(
        n_components=vocab_sz,
        random_state=idx,
        params='e',
        init_params='e'
        )
    
    model.n_features = vocab_sz
    
    # set startprob to alawys start on first state
    # model.startprob_ = np.zeros(plaintext_vocab_sz)
    # model.startprob_[0] = 1
    # set starprob to uniform
    model.startprob_ = np.ones(vocab_sz) / vocab_sz
    
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

print(f'Best score:      {best_score}')        
# use the Viterbi algorithm to predict the most likely sequence of states
# given the model
states = best_model.predict(X_train)
print(f"{states=}")

deciphered = [ct2pt[vocab[i]] for i in states]
print(f"{deciphered=}")