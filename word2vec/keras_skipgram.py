import numpy as np
import tensorflow as tf
import utils
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import time
from collections import defaultdict, Counter
import random
from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
import keras
from sklearn.model_selection import train_test_split

import urllib
import collections
import os
import zipfile


dataset_folder_path = 'data'
dataset_filename = 'text8.zip'
dataset_name = 'Text8 Dataset'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def get_unk(word, vocab):
    return word if word in vocab else 'UNK'

# ntlk.download(): stopwords, punkt
def preprocessing(text, vocab_size):
    start_time = time.time()
    stop_words = set(stopwords.words('english'))
    punctuations = set(string.punctuation)
    exclusions = stop_words.union(punctuations)
    token = word_tokenize(text)
    processed = [word.lower()
                 for word in token
                 if word.lower() not in exclusions]
    vocab = [word_count[0]
                for word_count in Counter(processed).most_common(vocab_size-1)]
    if vocab_size < len(processed):
        vocab.insert(0, 'UNK')
    processed = [get_unk(word, vocab) for word in processed]
    word_count = Counter(processed)
    print('Preprocessing done in %s seconds' % ((time.time() - start_time)))
    return processed, word_count, vocab

def get_lookup_tables(vocab):
    word_index = {}
    index_word = {}
    for index, word in enumerate(vocab):
        word_index.setdefault(word, index)
        index_word.setdefault(index, word)
    return word_index, index_word

def get_indexed_text(text, word_index):
    indexed_text =  [word_index[word] for word in text]
    return indexed_text

def build_data_set(text, vocab_size):
    processed, word_count, vocab = preprocessing(text, vocab_size)
    word_index, index_word = get_lookup_tables(vocab)
    indexed_text = get_indexed_text(processed, word_index)
    del vocab
    return indexed_text, word_count, word_index, index_word

def get_subSample(text):
    threshold = 1e-5 #how?
    all_count = len(text)
    word_freq = {word: count/all_count
                 for word, count in Counter(text).items()}
    p_drop = {word: 1-np.sqrt(threshold/word_freq[word])
              for word in indexed_text}
    subSample = [word for word in text if random.random() < (1 - p_drop[word])]
    return subSample

def get_surroundings(text, index, window_size=5):
    roll = np.random.randint(1, window_size+1)
    text_length = len(text)
    start = index - roll if (index - roll) > 0 else 0
    end = index + roll if (index + roll) < text_length else text_length
    surroundings = text[start:index] + text[index+1:end+1]
    return surroundings

def get_batches(text, batch_size, window_size=5):
    n_batch = len(text)//batch_size
    text = text[: n_batch*batch_size]
    for index in range(0, len(text), batch_size):
        X, Y = [], []
        batch = text[index: index+batch_size]
        for i in range (len(batch)):
            x = batch[i]
            y = get_surroundings(batch, i, window_size)
            X.append(x)
            Y.extend(y)
        yield X, Y

# if not isfile(dataset_filename):
#     with DLProgress(unit='B', unit_scale=True, miniters=1, desc=dataset_name) as pbar:
#         urlretrieve(
#             'http://mattmahoney.net/dc/text8.zip',
#             dataset_filename,
#             pbar.hook)
#
# if not isdir(dataset_folder_path):
#     with zipfile.ZipFile(dataset_filename) as zip_ref:
#         zip_ref.extractall(dataset_folder_path)
#
# with open('./data/text8') as f:
#     raw_text = f.read()

from nltk.corpus import gutenberg
raw_text = gutenberg.raw('melville-moby_dick.txt')

vocab_size = 1000
indexed_text, word_count, word_index, index_word = build_data_set(raw_text, vocab_size)
window_size = 5
vec_dim = 100
epochs = 5
valid_size = 16
valid_window = 100
valid_sample = np.random.choice(valid_window, valid_size, replace=False)

# Subsampling word from text
# i.e. word_sample_table[i] is the prob. of sampling ith most common word
# more common -> lower prob.
word_sample_table = sequence.make_sampling_table(vocab_size)

word_pairs, labels = skipgrams(indexed_text, vocab_size, shuffle=True,
                            window_size=window_size, sampling_table=word_sample_table)




X_train, X_heldout, y_train, y_heldout = train_test_split(word_pairs, labels, test_size=0.3, random_state=1)
model = keras.models.Sequential()
model.add(Embedding(vocab_size, vec_dim, input_length=2))
model.add(keras.layers.Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_heldout, y_heldout),
          batch_size=10, epochs=epochs, verbose=2)


## ref http://adventuresinmachinelearning.com/word2vec-keras-tutorial/

# # create some input variables
word_target, word_context = zip(*word_pairs)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")
input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vec_dim, input_length=1, name='embedding')
target = embedding(input_target)
target = Reshape((vec_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vec_dim, 1))(context)
#
# # setup a cosine similarity operation which will be output in a secondary model
# similarity = merge([target, context], mode='cos', dot_axes=0)
#
# # now perform the dot product operation to get a similarity measure
# dot_product = merge([target, context], mode='dot', dot_axes=1)
# dot_product = Reshape((1,))(dot_product)
# # add the sigmoid output layer
# output = Dense(1, activation='sigmoid')(dot_product)
# # create the primary training model
# model = Model(input=[input_target, input_context], output=output)
# model.compile(loss='binary_crossentropy', optimizer='rmsprop')
#
# validation_model = Model(input=[input_target, input_context], output=similarity)
#
#
# class SimilarityCallback:
#     def run_sim(self):
#         for i in range(valid_size):
#             valid_word = index_word[valid_sample[i]]
#             top_k = 8  # number of nearest neighbors
#             sim = self._get_sim(valid_sample[i])
#             nearest = (-sim).argsort()[1:top_k + 1]
#             log_str = 'Nearest to %s:' % valid_word
#             for k in range(top_k):
#                 close_word = index_word[nearest[k]]
#                 log_str = '%s %s,' % (log_str, close_word)
#             print(log_str)
#
#     @staticmethod
#     def _get_sim(valid_word_idx):
#         sim = np.zeros((vocab_size,))
#         in_arr1 = np.zeros((1,))
#         in_arr2 = np.zeros((1,))
#         in_arr1[0,] = valid_word_idx
#         for i in range(vocab_size):
#             in_arr2[0,] = i
#             out = validation_model.predict_on_batch([in_arr1, in_arr2])
#             sim[i] = out
#         return sim
# sim_cb = SimilarityCallback()
#
# arr_1 = np.zeros((1,))
# arr_2 = np.zeros((1,))
# arr_3 = np.zeros((1,))
# for cnt in range(epochs):
#     idx = np.random.randint(0, len(labels)-1)
#     arr_1[0,] = word_target[idx]
#     arr_2[0,] = word_context[idx]
#     arr_3[0,] = labels[idx]
#     loss = model.train_on_batch([arr_1, arr_2], arr_3)
#     if cnt % 100 == 0:
#         print("Iteration {}, loss={}".format(cnt, loss))
#     if cnt % 10000 == 0:
#         sim_cb.run_sim()
