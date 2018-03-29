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
import jellyfish
import gc
import itertools
import pandas as pd
import collections as col
from collections import Counter 
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers.core import Masking
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling1D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Conv1D, MaxPool1D, Conv2D, MaxPool2D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from sklearn.cross_validation import StratifiedKFold
from nltk import tokenize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from attention import AttLayer
from svmwrapper import SVMWrapper
from lmwrapper import LMWrapper

# Stopping Criteria for training the model
earlyStopping = EarlyStopping(monitor = 'loss', min_delta = 0.3, patience=1, verbose=0, mode='auto')          

# Set parameters:
max_features = 50000            # Maximum number of tokens in vocabulary
maxlen = 400                    # Maximum Length of each Sentence
maxsents = 9                    # Maximum Number of Sentences (5 for Death Certificate + 1 for Autopsy Report + 1 for Clinical Information Bulletin)
maxsents_co = 5                 # Number of Sentences in Death Certificate
batch_size = 32                 # Batch size given to the model while training
embedding_dims = 175            # Embedding Dimensions
maxchars_word = 50              # Maximum number of chars in each token
nb_epoch = 50                   # Number of epochs for training
validation_split = 0.25         # Percentage of the dataset used in validation                                                         
gru_output_size = 175           # GRU output dimension

print('Loading data...')
# Shape of each line in dataset:
# 'Full ICD-10 code of underlying death cause' <> 'Death Certificate' <> 'Clinical Information Bulletin' <> 'Autopsy Report' <> 'Full ICD-10 codes present in Death Certificate'
texts = [ line.rstrip('\n') for line in codecs.open('example_dataset.txt', encoding="utf-8") ]                                                    

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
    labels_aux[i,:] = sum( to_categorical(labels_int_aux[i],num_classes))
    
num_classes_3=1+max([max(x) for x in labels_int_3_aux])    
labels_3_aux = np.zeros((len(labels), num_classes_3), dtype=np.float64)
for i in range(len(labels_int_3_aux)):
    labels_3_aux[i,:] = sum( to_categorical(labels_int_3_aux[i],num_classes_3))

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

#%%
case2Idx = {'numeric': 1, 'allLower':2, 'allUpper':3, 'initialUpper':4, 'other':5, 'mainly_numeric':6, 'contains_digit': 7, 'PADDING_TOKEN':0}
caseEmbeddings = np.identity(len(case2Idx), dtype='float32')
char2Idx = {"PADDING":0, "UNKNOWN":1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzàáâãéêíóôõúüABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÉÊÍÓÔÕÚÜ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|": char2Idx[c] = len(char2Idx)

def getCasing(word, caseLookup):   
    casing = 'other'
    numDigits = 0
    for char in word:
        if char.isdigit(): numDigits += 1
    digitFraction = numDigits / float(len(word))
    if word.isdigit(): casing = 'numeric'
    elif digitFraction > 0.5: casing = 'mainly_numeric'
    elif word.islower(): casing = 'allLower'
    elif word.isupper(): casing = 'allUpper'
    elif word[0].isupper(): casing = 'initialUpper'
    elif numDigits > 0: casing = 'contains_digit'
    return caseLookup[casing]

def getChars(word, charLookup):
    aux = [ charLookup["PADDING"] ]
    for char in word:
        if char in charLookup: aux.append(charLookup[char])
        else: aux.append(charLookup["UNKNOWN"])
    while len(aux) < maxchars_word: aux.append(charLookup["PADDING"])
    return np.array( aux )

# attribute an integer to each token that occures in the texts 
# conversion of each dataset entry in a (7,200) shape matrix resulting in variables:

print('Computing Training Set...')

X_train = np.zeros((len(X_train_1a), maxsents, maxlen), dtype = 'int32')
X_train_char = np.zeros((len(X_train_1a), maxsents, maxlen, maxchars_word), dtype = 'int32')
X_train_casing = np.zeros((len(X_train_1a), maxsents, maxlen), dtype = 'int32')

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
                        X_train_casing[i,m,k] = getCasing(word,case2Idx)
                        X_train_char[i,m,k,:] = getChars(word,char2Idx)
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
                        X_train_casing[i,5+m,k] = getCasing(word,case2Idx)
                        X_train_char[i,5+m,k,:] = getChars(word,char2Idx)
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
                    X_train_casing[i,8,k] = getCasing(word,case2Idx)
                    X_train_char[i,8,k,:] = getChars(word,char2Idx)
                    k = k + 1

word_index = tokenizer.word_index

np.save('DICT.npy', word_index)
np.save('FULL_CODES.npy', le4)
np.save('BLOCKS.npy', le3)

print('Found %s unique tokens.' % len(word_index))


#%%
print('Computing Testing Set...')

X_test = np.zeros((len(X_test_1a), maxsents, maxlen), dtype = 'int32')
X_test_char = np.zeros((len(X_test_1a), maxsents, maxlen, maxchars_word), dtype = 'int32')
X_test_casing = np.zeros((len(X_test_1a), maxsents, maxlen), dtype = 'int32')

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
                        X_test_casing[i,m,k] = getCasing(word,case2Idx)
                        X_test_char[i,m,k,:] = getChars(word,char2Idx)
                        k = k + 1
                else:
                    if k < maxlen and word_index.get(word) < max_features:
                        X_test[i,m,k] = word_index.get(word)
                        X_test_casing[i,m,k] = getCasing(word,case2Idx)
                        X_test_char[i,m,k,:] = getChars(word,char2Idx)
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
                        X_test_casing[i,5+m,k] = getCasing(word,case2Idx)
                        X_test_char[i,5+m,k,:] = getChars(word,char2Idx)
                        k = k + 1
                else:
                    if k < maxlen and word_index.get(word) < max_features:
                        X_test[i,5+m,k] = word_index.get(word)
                        X_test_casing[i,5+m,k] = getCasing(word,case2Idx)
                        X_test_char[i,m,k,:] = getChars(word,char2Idx)
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
                    X_test_casing[i,8,k] = getCasing(word,case2Idx)
                    X_test_char[i,8,k,:] = getChars(word,char2Idx)
                    k = k + 1
            else:
                if k < maxlen and word_index.get(word) < max_features:
                    X_test[i,8,k] = word_index.get(word)
                    X_test_casing[i,8,k] = getCasing(word,case2Idx)
                    X_test_char[i,8,k,:] = getChars(word,char2Idx)
                    k = k + 1

#%%

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

#%%

print('Computing Initialization with Label Co-occurrence...')

train_labels_aux = [np.where(x != 0)[0] for x in y_train_aux]
train_labels_3_aux = [np.where(x != 0)[0] for x in y_train_3_aux]
train_labels_full = [np.where(x != 0)[0] for x in y_train]
train_labels_3char = [np.where(x != 0)[0] for x in y_train_3char]

train_labels_aux_cid = []
train_labels_full_cid = []

train_labels_3_aux_cid = []
train_labels_3char = []

# 4char
for i in range(len(train_labels_full)):
    train_labels_full_cid.extend(list(le4.inverse_transform(train_labels_full[i])))

for i in range(len(train_labels_aux)):
    train_labels_aux_cid.append(list(le4_aux.inverse_transform(train_labels_aux[i])))

# 3char
for i in range(len(train_labels_3char)):
    train_labels_3char.extend(list(le3.inverse_transform(train_labels_3char[i])))

for i in range(len(train_labels_3_aux)):
    train_labels_3_aux_cid.append(list(le3_aux.inverse_transform(train_labels_3_aux[i])))

# 4char
extra_labels_cid = np.setdiff1d([item for sublist in labels_cid_aux for item in sublist],labels_cid)
extra_labels = le4_aux.transform(extra_labels_cid)

common_labels_cid = list(set([item for sublist in labels_cid_aux for item in sublist]).intersection(labels_cid))
common_labels = le4_aux.transform(common_labels_cid)

# 3char
extra_labels_3_cid = np.setdiff1d([item for sublist in labels_cid_3_aux for item in sublist],labels_cid_3char)
extra_labels_3 = le3_aux.transform(extra_labels_3_cid)

common_labels_3_cid = list(set([item for sublist in labels_cid_3_aux for item in sublist]).intersection(labels_cid_3char))
common_labels_3 = le4_aux.transform(common_labels_3_cid)

#%%
# 4char
init_m_aux = np.zeros((num_classes,num_classes), dtype=np.float32)
bias_aux = np.zeros((num_classes,1), dtype=np.float32)

for i in range(len(train_labels_aux)):
    for j in range(len(train_labels_aux[i])):
        for k in range(len(train_labels_aux[i])):
            init_m_aux[train_labels_aux[i][j],train_labels_aux[i][k]] += 1

nmf = NMF(n_components=gru_output_size+embedding_dims)
init_m_aux = np.log2(init_m_aux + 1)
nmf.fit(init_m_aux)
init_m_aux = nmf.components_
   
init_m_full = np.zeros((y_train.shape[1],y_train.shape[1]))
bias_full = np.zeros((y_train.shape[1],1))

for i in range(len(train_labels_aux)):
    row = [x for x in train_labels_aux_cid[i] if x in common_labels_cid]
    for j in row:
        for k in row:
            a = le4.transform([j])
            b = le4.transform([k])
            init_m_full[a,b] += 1

nmf = NMF(n_components=gru_output_size+embedding_dims)
init_m_full = np.log2(init_m_full + 1)
nmf.fit(init_m_full)
init_m_full = nmf.components_


#%%
# 3char
init_m_3 = np.zeros((y_train_3char.shape[1],y_train_3char.shape[1]))
bias_3 = np.zeros((y_train_3char.shape[1],1))

for i in range(len(train_labels_3_aux)):
    row = [x for x in train_labels_3_aux_cid[i] if x in common_labels_3_cid]
    for j in row:
        for k in row:
            a = le3.transform([j])
            b = le3.transform([k])
            init_m_3[a,b] += 1

nmf = NMF(n_components=gru_output_size+embedding_dims)
init_m_3 = np.log2(init_m_3 + 1)
nmf.fit(init_m_3)
init_m_3 = nmf.components_

#%%
print('Build model...')

# Inputs
review_input_words = Input(shape=(maxsents,maxlen), dtype='int32')
review_input_casing = Input(shape=(maxsents,maxlen), dtype='int32')
review_input_chars = Input(shape=(maxsents,maxlen,maxchars_word), dtype='int32')
aux_input = Input(shape=(y_train.shape[1],), dtype='float32')

# Embedding Layers
embedding_layer_words = Embedding(max_features, embedding_dims, input_length=maxlen)
embedding_layer_casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False)
embedding_layer_character = Embedding(len(char2Idx),25,embeddings_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5))

# WORD-LEVEL AND CHARACTER-LEVEL
sentence_input_words = Input(shape=(maxlen,), dtype='int32')
embedded_sequences_words = embedding_layer_words(sentence_input_words)
sentence_input_casing = Input(shape=(maxlen,), dtype='int32')
embedded_sequences_casing = embedding_layer_casing(sentence_input_casing)
sentence_input_chars = Input(shape=(maxlen,maxchars_word), dtype='int32')
embedded_sequences_chars = TimeDistributed(embedding_layer_character)(sentence_input_chars)
embedded_sequences_chars = TimeDistributed(Bidirectional(GRU(embedding_dims, return_sequences=False)))(embedded_sequences_chars)

review_words = TimeDistributed(Model(sentence_input_words, embedded_sequences_words))(review_input_words)
review_casing = TimeDistributed(Model(sentence_input_casing, embedded_sequences_casing))(review_input_casing)
review_chars = TimeDistributed(Model(sentence_input_chars, embedded_sequences_chars))(review_input_chars)
review_embedded = keras.layers.Concatenate()( [ review_words , review_casing , review_chars] )
review_embedded = TimeDistributed(TimeDistributed(Dense(embedding_dims,activation='tanh')))(review_embedded)

# Embedding Average or Convolutional Neural Network
#fasttext = GlobalAveragePooling2D()(review_embedded)
reshape = Reshape((maxsents*maxlen,embedding_dims))(review_embedded)
conv1 = Conv1D(gru_output_size, kernel_size=2, activation='tanh')(reshape)
conv1 = MaxPool1D(int((maxsents*maxlen)/5))(conv1)
conv1 = Flatten()(conv1)
conv2 = Conv1D(gru_output_size, kernel_size=4, activation='tanh')(reshape)
conv2 = MaxPool1D(int((maxsents*maxlen)/5))(conv2)
conv2 = Flatten()(conv2)
conv3 = Conv1D(gru_output_size, kernel_size=8, activation='tanh')(reshape)
conv3 = MaxPool1D(int((maxsents*maxlen)/5))(conv3)
conv3 = Flatten()(conv3)
concatenated_tensor = keras.layers.Concatenate(axis=1)([conv1,conv2,conv3])
fasttext = Dense(units=gru_output_size, activation='tanh')(concatenated_tensor)

# Bidirectional GRU
l_gru_sent = TimeDistributed(Bidirectional(GRU(gru_output_size, return_sequences=True)))(review_embedded)
l_dense_sent = TimeDistributed(TimeDistributed(Dense(units=gru_output_size)))(l_gru_sent)
l_att_sent = TimeDistributed(AttLayer())(l_dense_sent)

# Bidirectional GRU
l_gru_review = Bidirectional(GRU(gru_output_size, return_sequences=True))(l_att_sent)
l_gru_review = keras.layers.Concatenate()( [ l_gru_review , keras.layers.RepeatVector(maxsents)(fasttext) ] )
l_dense_review = TimeDistributed(Dense(units=gru_output_size))(l_gru_review)
l_att_review = AttLayer()(l_dense_review)

#Memory
aux_mem = Dense(units=(gru_output_size+embedding_dims), activation='tanh', weights=(init_m_full.transpose(),np.zeros(gru_output_size+embedding_dims)), name='memory')(aux_input)
postp = keras.layers.Concatenate( axis = 1 )( [ l_att_review , fasttext , aux_mem ] )
postp = Dropout(0.05)(postp)
postp = Dense(units=(gru_output_size+embedding_dims))(postp)

# Softmax
preds = Dense(units=y_train.shape[1], activation='softmax', weights=(init_m_full,np.zeros(y_train.shape[1])), name='full_code')(postp)
preds_3char = Dense(units=y_train_3char.shape[1], activation='softmax', weights=(init_m_3,np.zeros(y_train_3char.shape[1])), name='block')(postp)
preds_aux = Dense(units=y_train_aux.shape[1], activation='sigmoid', weights=(init_m_aux,np.zeros(y_train_aux.shape[1])), name='aux')(postp)

model = LMWrapper()
model.fit(X_train, y_train, validation_data=(X_test,y_test))
model.save('modelo_baseline.h5')
X_train_aux = model.predict_prob(X_train)[0]
X_test_aux = model.predict_prob(X_test)[0]

model = Model(inputs=[review_input_words, review_input_casing, review_input_chars, aux_input], outputs = [preds, preds_3char, preds_aux])
model.compile(loss=['categorical_crossentropy','categorical_crossentropy','binary_crossentropy'], optimizer='adam', metrics=['accuracy'], loss_weights = [0.8 , 0.85, 0.75])
model.summary()
model.fit([ X_train , X_train_casing, X_train_char, X_train_aux ] , [y_train, y_train_3char, y_train_aux], batch_size=batch_size, epochs=nb_epoch, validation_data=([X_test,X_test_casing,X_test_char,X_test_aux], [y_test, y_test_3char, y_test_aux]), callbacks=[earlyStopping])
model.save('modelo_full_nmf.h5')

print('Predicting...')
[all_4, all_3, all_aux] = model.predict([X_test,X_test_casing,X_test_char,X_test_aux], batch_size=3)

print('Writing output...')

cid_pred = np.zeros([len(y_test),7], dtype = object)

for i in range(len(y_test)):
    top3_4 = np.argsort(all_4[i])[-3:]
    top3_3 = np.argsort(all_3[i])[-3:]
    cid_pred[i][0] = le4.inverse_transform(np.argmax(y_test[i]))
    for j in [1,2,3]:
        cid_pred[i][j] = le4.inverse_transform(top3_4[-j])
        cid_pred[i][3+j] = le3.inverse_transform(top3_3[-j])

np.savetxt('pred_full_nmf.txt', cid_pred, delimiter=" ", fmt="%s")
