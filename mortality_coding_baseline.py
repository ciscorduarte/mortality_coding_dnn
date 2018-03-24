#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)
    warnings.filterwarnings("ignore",category=UserWarning)
    import sklearn
    import h5py     
    import keras
import os
import codecs
import theano
import gc
import itertools
import sklearn
import jellyfish
import collections as col
import pandas as pd
from collections import Counter 
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from sklearn.cross_validation import StratifiedKFold
from nltk import tokenize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from svmwrapper import SVMWrapper
from lmwrapper import LMWrapper

# Set parameters:
max_features = 50000            # Maximum number of tokens in vocabulary
maxlen = 400                    # Maximum Length of each Sentence
maxsents = 9                    # Maximum Number of Sentences (5 for Death Certificate + 1 for Autopsy Report + 1 for Clinical Information Bulletin)
maxsents_co = 5                 # Number of Sentences in Death Certificate
batch_size = 32                 # Batch size given to the model while training
embedding_dims = 175            # Embedding Dimensions
nb_epoch = 50                   # Number of epochs for training
validation_split = 0.25         # Percentage of the dataset used in validation                                                         
gru_output_size = 175           # GRU output dimension
kernel_size = 5
filters = 50
pool_size = 4

print('Loading data...')
# Shape of each line in dataset:
# 'Full ICD-10 code of underlying death cause' <> 'Death Certificate' <> 'Clinical Information Bulletin' <> 'Autopsy Report' <> 'Full ICD-10 codes present in Death Certificate'
texts = [ line.rstrip('\n') for line in codecs.open('example_dataset.txt', 
         encoding="utf-8") ]                                                    

# labels_cid is a list of the ICD-10 full code for the underlying death cause for each dataset entry
labels_cid = list([ line.split('<>')[0][:-1] for line in texts ])

# labels_cid_3char is identic to labels_cid but the code is truncated to 3 characters (ICD-10 block)
labels_cid_3char = [x[:3] for x in labels_cid]

# labels_cid_aux is a list of the ICD-10 full codes present in the death certificate
labels_cid_aux = [ line.split('<>')[10].replace("'","") for line in texts ]
labels_cid_aux = [x[2:-2] for x in labels_cid_aux]
labels_cid_aux = [x.split(', ') for x in labels_cid_aux]

labels_cid_3_aux = labels_cid_aux
for i in range(len(labels_cid_3_aux)):
    labels_cid_3_aux[i] = [x[:3] for x in labels_cid_aux[i]]

# Using sklearn package attribute an integer to each code that occures resulting in the variables:
# labels_int, labels_int_3char, labels_int_aux 
le3 = preprocessing.LabelEncoder()
le4 = preprocessing.LabelEncoder()
le4_aux = preprocessing.LabelEncoder()
le3_aux = preprocessing.LabelEncoder()

char3 = le3.fit(labels_cid_3char)           
char4 = le4.fit(labels_cid)
char4_aux = le4_aux.fit([item for sublist in labels_cid_aux for item in sublist])
char3_aux = le3_aux.fit([item for sublist in labels_cid_3_aux for item in sublist])

labels_int_3char = char3.transform(labels_cid_3char)
labels_int = char4.transform(labels_cid)

labels_int_aux = np.copy(labels_cid_aux)
labels_int_3_aux = np.copy(labels_cid_3_aux)

for i in range(len(labels_int_aux)):
    labels_int_aux[i] = char4_aux.transform(labels_int_aux[i])
    labels_int_3_aux[i] = char3_aux.transform(labels_int_3_aux[i])

part_1a = [ line.split('<>')[1].lower() for line in texts ]
part_1b = [ line.split('<>')[2].lower() for line in texts ]
part_1c = [ line.split('<>')[3].lower() for line in texts ]
part_1d = [ line.split('<>')[4].lower() for line in texts ]
part_2 = [ line.split('<>')[5].lower() for line in texts ]
bic = [ line.split('<>')[6].lower() for line in texts ]
bic_admiss = [ line.split('<>')[7].lower() for line in texts ]
bic_sit = [ line.split('<>')[8].lower() for line in texts ]
ra = [ line.split('<>')[9].lower() for line in texts ]

labels_int = np.asarray(labels_int)
labels_int_aux = np.asarray(labels_int_aux)
labels_int_3char = np.asarray(labels_int_3char)

# Conversion of the Full ICD-10 code into a one-hot vector
# e.g. J189 (in labels_cid) -> 3 (in labels_int) -> [0, 0, 0, 1, 0, (...), 0] (in labels)

labels = to_categorical(labels_int)                
labels_3char = to_categorical(labels_int_3char)

num_classes=1+max([max(x) for x in labels_int_aux])    
labels_aux = np.zeros((len(labels), num_classes), dtype=np.float64)
for i in range(len(labels_int_aux)):
    labels_aux[i,:] = sum( to_categorical(labels_int_aux[i], num_classes=num_classes))
    
num_classes_3=1+max([max(x) for x in labels_int_3_aux])    
labels_3_aux = np.zeros((len(labels), num_classes_3), dtype=np.float64)
for i in range(len(labels_int_3_aux)):
    labels_3_aux[i,:] = sum( to_categorical(labels_int_3_aux[i], num_classes=num_classes_3))

#%%

print('Spliting the data into a training set and a validation set...')

X_train_1a, X_test_1a, y_train, y_test = train_test_split(part_1a, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1a, X_test_1a, y_train_aux, y_test_aux = train_test_split(part_1a, labels_aux, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1a, X_test_1a, y_train_3char, y_test_3char = train_test_split(part_1a, labels_3char, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1a, X_test_1a, y_train_3_aux, y_test_3_aux = train_test_split(part_1a, labels_3_aux, stratify = labels_cid, test_size = 0.25, random_state=42)

X_train_1b, X_test_1b, y_train, y_test = train_test_split(part_1b, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1c, X_test_1c, y_train, y_test = train_test_split(part_1c, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1d, X_test_1d, y_train, y_test = train_test_split(part_1d, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_2, X_test_2, y_train, y_test = train_test_split(part_2, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_bic, X_test_bic, y_train, y_test = train_test_split(bic, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_bic_admiss, X_test_bic_admiss, y_train, y_test = train_test_split(bic_admiss, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_bic_sit, X_test_bic_sit, y_train, y_test = train_test_split(bic_sit, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_ra, X_test_ra, y_train, y_test = train_test_split(ra, labels, stratify = labels_cid, test_size = 0.25, random_state=42)

#%%
tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(X_train_1a+X_train_1b+X_train_1c+X_train_1d+X_train_2+X_train_bic+X_train_bic_admiss+X_train_bic_sit+X_train_ra)

# attribute an integer to each token that occures in the texts 
# conversion of each dataset entry in a (7,200) shape matrix resulting in variables:

print('Computing Training Set...')

# data is a (5,200) matrix for the strings in death certificates
X_train = np.zeros((len(X_train_1a), maxsents, maxlen), dtype = 'int32')

print('Loading death certificates...')

death_cert = [X_train_1a, X_train_1b, X_train_1c, X_train_1d, X_train_2]
for m in range(len(death_cert)):
    part = death_cert[m]
    for i, sentences in enumerate(part):
        sentences = tokenize.sent_tokenize( sentences )
        k = 0
        for j, sent in enumerate(sentences):
            if j < maxsents:
                wordTokens = text_to_word_sequence(sent)
                for _ , word in enumerate(wordTokens):
                    if k < maxlen and tokenizer.word_index[word] < max_features:
                        X_train[i,m,k] = tokenizer.word_index[word]
                        k = k + 1
                    
print('Loading bic...')

bic_components = [X_train_bic, X_train_bic_admiss, X_train_bic_sit]
for m in range(len(bic_components)):
    bic_part = bic_components[m]
    for i, sentences in enumerate(bic_part):
        sentences = tokenize.sent_tokenize( sentences )
        k = 0
        for j, sent in enumerate(sentences):
            if j < maxsents:
                wordTokens = text_to_word_sequence(sent)
                for _ , word in enumerate(wordTokens):
                    if k < maxlen and tokenizer.word_index[word] < max_features:
                        X_train[i,5+m,k] = tokenizer.word_index[word]
                        k = k + 1

print('Loading autopsy reports...')

for i, sentences in enumerate(X_train_ra):
    sentences = tokenize.sent_tokenize( sentences )
    k = 0
    for j, sent in enumerate(sentences):
        if j < maxsents:
            wordTokens = text_to_word_sequence(sent)
            for _ , word in enumerate(wordTokens):
                if k < maxlen and tokenizer.word_index[word] < max_features:
                    X_train[i,8,k] = tokenizer.word_index[word]
                    k = k + 1

word_index = tokenizer.word_index

np.save('DICT.npy', word_index)
np.save('FULL_CODES.npy', le4)
np.save('BLOCKS.npy', le3)

print('Found %s unique tokens.' % len(word_index))


#%%
print('Computing Testing Set...')

X_test = np.zeros((len(X_test_1a), maxsents, maxlen), dtype = 'int32')

print('Loading Death certificates...')

death_cert = [X_test_1a, X_test_1b, X_test_1c, X_test_1d, X_test_2]

for m in range(len(death_cert)):
    part = death_cert[m]
    for i, sentences in enumerate(part):
        sentences = tokenize.sent_tokenize( sentences )
        k = 0
        for j, sent in enumerate(sentences):
            wordTokens = text_to_word_sequence(sent)
            for _ , word in enumerate(wordTokens):
                if word_index.get(word) == None: 
                    aux = [(jellyfish.jaro_winkler(k,word),v) for k,v in word_index.items()]
                    if k < maxlen and max(aux)[1] < max_features:
                        X_test[i,m,k] = max(aux)[1]
                        k = k + 1
                else:
                    if k < maxlen and word_index.get(word) < max_features:
                        X_test[i,m,k] = word_index.get(word)
                        k = k + 1
                    
print('Loading bic...')

bic_components = [X_test_bic, X_test_bic_admiss, X_test_bic_sit]
for m in range(len(bic_components)):
    bic_part = bic_components[m]
    for i, sentences in enumerate(bic_part):
        sentences = tokenize.sent_tokenize( sentences )
        k = 0
        for j, sent in enumerate(sentences):
            wordTokens = text_to_word_sequence(sent)
            for _ , word in enumerate(wordTokens):
                if word_index.get(word) == None: 
                    aux = [(jellyfish.jaro_winkler(k,word),v) for k,v in word_index.items()]
                    if k < maxlen and max(aux)[1] < max_features:
                        X_test[i,5+m,k] = max(aux)[1]
                        k = k + 1
                else:
                    if k < maxlen and word_index.get(word) < max_features:
                        X_test[i,5+m,k] = word_index.get(word)
                        k = k + 1

print('Loading autopsy reports...')

for i, sentences in enumerate(X_test_ra):
    sentences = tokenize.sent_tokenize( sentences )
    k = 0
    for j, sent in enumerate(sentences):
        wordTokens = text_to_word_sequence(sent)
        for _ , word in enumerate(wordTokens):
            if word_index.get(word) == None: 
                aux = [(jellyfish.jaro_winkler(k,word),v) for k,v in word_index.items()]
                if k < maxlen and max(aux)[1] < max_features:
                    X_test[i,8,k] = max(aux)[1]
                    k = k + 1
            else:
                if k < maxlen and word_index.get(word) < max_features:
                    X_test[i,8,k] = word_index.get(word)
                    k = k + 1

#%%

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
#model = SVMWrapper()
model = LMWrapper()

#%%
model.fit(X_train, y_train, validation_data=(X_test,y_test))
model.save('modelo_baseline.h5')

print('Predicting...')
all_4 = model.predict(X_test)[0]

print('Writing output...')

cid_pred = np.zeros([len(y_test),7], dtype = object)

for i in range(len(y_test)):
    top3_4 = np.argsort(all_4[i])[-3:]
    cid_pred[i][0] = le4.inverse_transform(np.argmax(y_test[i]))
    for j in [1,2,3]: cid_pred[i][j] = le4.inverse_transform(top3_4[-j])

np.savetxt('pred_baseline.txt', cid_pred, delimiter=" ", fmt="%s")
