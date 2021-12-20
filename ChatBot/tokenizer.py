import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence):
    sentence = nltk.word_tokenize(sentence)
    return [stemmer.stem(word.lower()) for word in sentence]

def bagOfWords(tked_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, w in enumerate(all_words):
        if w in tked_sentence:
            bag[i] = 1.0

    return bag
