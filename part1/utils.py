import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
import re
import os
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # Transformation: "noisy user text" = noun synonym replacement + keyboard typos.
    #  1. each word with POS tag SYN_TAG (JJ = adjective, NN = noun; set via env SYN_POS) is replaced, with probability P_SYN, by a WordNet synonym
    #     taken from the noun's most frequent sense only (so "movie" -> "film"-like words, not wrong-sense words)
    #  2. each remaining eligible word gets one letter replaced by a neighbouring QWERTY key
    #     with probability P_TYPO
    # Negations and short words (<= 3 letters) are never touched, so the label should not change.
    tagged = pos_tag(word_tokenize(example["text"]))
    new_tokens = []
    for tok, tag in tagged:
        eligible = tok.isalpha() and len(tok) > 3 and tok.lower() not in NEGATIONS
        if eligible and tag == SYN_TAG and random.random() < P_SYN:
            syns = _synonyms(tok)
            if syns:
                new_tokens.append(random.choice(syns))
                continue
        if eligible and random.random() < P_TYPO:
            tok = _typo(tok)
        new_tokens.append(tok)
    text = TreebankWordDetokenizer().detokenize(new_tokens)
    example["text"] = re.sub(r"\s+([.,!?;:])", r"\1", text)

    ##### YOUR CODE ENDS HERE ######

    return example


# --- helpers / knobs for custom_transform (tune P_SYN / P_TYPO to change the accuracy drop) ---
P_SYN = 0.5
SYN_POS = os.environ.get("SYN_POS", "noun")  # "noun" (default) or "adj"; lets us compare the two
SYN_TAG = "JJ" if SYN_POS == "adj" else "NN"
WN_POS = wordnet.ADJ if SYN_POS == "adj" else wordnet.NOUN
P_TYPO = float(os.environ.get("P_TYPO", "0.5"))   # chance of one QWERTY-neighbour typo per eligible word
NEGATIONS = {"not", "never", "none", "nothing", "nobody", "neither", "nor", "without", "cannot"}
_QWERTY = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
    "a": "qsz", "s": "awdx", "d": "sefc", "f": "drgv", "g": "fthb", "h": "gyjn", "j": "hukm", "k": "jilm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}


def _synonyms(word):
    out = set()
    for syn in wordnet.synsets(word.lower(), pos=WN_POS)[:1]:
        for lemma in syn.lemmas():
            name = lemma.name()
            if "_" not in name and name.isalpha() and name.lower() != word.lower():
                out.add(name)
    return sorted(out)


def _typo(word):
    positions = [i for i, c in enumerate(word) if c.lower() in _QWERTY]
    if not positions:
        return word
    i = random.choice(positions)
    repl = random.choice(_QWERTY[word[i].lower()])
    if word[i].isupper():
        repl = repl.upper()
    return word[:i] + repl + word[i + 1:]
