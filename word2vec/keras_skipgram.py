import string
import time
import keras
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
from collections import Counter
from keras.models import Model
from keras.layers import Input, Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import skipgrams


# class DLProgress(tqdm):
#     last_block = 0
#
#     def hook(self, block_num=1, block_size=1, total_size=None):
#         self.total = total_size
#         self.update((block_num - self.last_block) * block_size)
#         self.last_block = block_num


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
             for word_count in Counter(processed).most_common(vocab_size - 1)]
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
    indexed_text = [word_index[word] for word in text]
    return indexed_text


def build_data_set(text, vocab_size):
    processed, word_count, vocab = preprocessing(text, vocab_size)
    word_index, index_word = get_lookup_tables(vocab)
    indexed_text = get_indexed_text(processed, word_index)
    del vocab
    return indexed_text, word_count, word_index, index_word

# dataset_folder_path = 'data'
# dataset_filename = 'text8.zip'
# dataset_name = 'Text8 Dataset'

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

# with open('./word2vec/data/text8') as f:
#     raw_text = f.read()

# from nltk.corpus import gutenberg
# raw_text = gutenberg.raw('austen-emma.txt')

from nltk.corpus import brown


vocab_size = 3000
window_size = 5
vec_dim = 300
epochs = 10
batch_size = 512

raw_text = ' '.join(brown.words()[:])
indexed_text, word_count, word_index, index_word = build_data_set(raw_text, vocab_size)


# Subsampling word from text
# i.e. word_sample_table[i] is the prob. of sampling ith most common word
# more common -> lower prob.
word_sample_table = sequence.make_sampling_table(vocab_size)

word_pairs, labels = skipgrams(indexed_text, vocab_size, shuffle=True, negative_samples=2,
                               window_size=window_size, sampling_table=word_sample_table)

word_target, word_context = zip(*word_pairs)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

input_target = Input((1,), name='target_input')
input_context = Input((1,), name='context_input')

embedding = Embedding(vocab_size, vec_dim, input_length=1, name='embedding')
target = embedding(input_target)
target = Reshape((vec_dim, 1), name='taget_reshape')(target)
context = embedding(input_context)
context = Reshape((vec_dim, 1), name='context_reshape')(context)

# # now perform the dot product operation to get a similarity measure
dot_product = keras.layers.dot([target, context], axes=1,
                               name='dot_product', normalize=True)
dot_product = Reshape((1,), name='dot_product_reshape')(dot_product)
# # add the sigmoid output layer
output = Dense(1, activation='sigmoid', name='sigmoid')(dot_product)
# # create the primary training model
model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
validation_model = Model(input=[input_target, input_context], output=dot_product)
vec_model = Model(input=input_target, output=target)

#
model.fit([word_target, word_context], labels,
          validation_split=0,
          batch_size=batch_size * 2, epochs=epochs, verbose=2)



def word2vec(word, vec_model, word_index=word_index):
    index = np.array([word_index[word]])
    return vec_model.predict_on_batch(index).flatten()


#
def get_k_most_common_context(target, validation_model, top_k,
                              lookups=(word_index, index_word), vocab_size=vocab_size):
    word_index, index_word = lookups
    target_indx = np.array([word_index[target]])
    similarities = np.zeros((vocab_size,))
    for i in range(vocab_size):
        context_indx = np.array([i])
        sim = validation_model.predict_on_batch([target_indx, context_indx])
        similarities[i] = sim
    nearest = (-similarities).argsort()[1:top_k + 1]
    out = [index_word[word] for word in nearest]
    print("%d Nearest to white %s :" % (top_k, target), out)
    return out

## ref http://adventuresinmachinelearning.com/word2vec-keras-tutorial/
## validation in process
# valid_size = 16
# valid_window = 100
# valid_sample = np.random.choice(valid_window, valid_size, replace=False)
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
#
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
