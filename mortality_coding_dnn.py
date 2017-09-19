from __future__ import print_function
import numpy as np
import os
import codecs
import theano
np.random.seed(1337) # for reproducibility
import collections as col
import gc
import itertools
import pandas as pd
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
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import merge
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializations
from sklearn.cross_validation import StratifiedKFold
from nltk import tokenize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from apriori import apriori

# Stopping Criteria for training the model
earlyStopping = EarlyStopping(monitor = 'loss', min_delta = 0.3, patience=1, verbose=0, mode='auto')          

# Attention Layer proposed by Yang et al. in 'Hierarchical Attention Networks for Document Classification' 
class AttLayer(Layer):    
    def __init__(self, **kwargs):
        self.init = initializations.get('normal')
        super(AttLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.init((input_shape[-1],))
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  
    
    def call(self, x, mask=None):
        
        eij = K.tanh(K.dot(x, self.W))    
        
        ai = K.exp(eij)    
        
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        
        self.att = weights
        # Attention weights are assigned to self.att to allow visualization 
        
        weighted_input = x*weights.dimshuffle(0,1,'x')  
                
        return weighted_input.sum(axis=1)
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Set parameters:
max_features = 35000            # Maximum number of tokens in vocabulary
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
texts = [ line.rstrip('\n') for line in codecs.open('DATASET.txt', 
         encoding="utf-8") ]                                                    

# labels_cid is a list of the ICD-10 full code for the underlying death cause for each dataset entry
labels_cid = list([ line.split('<>')[0][:-1] for line in texts ])

# labels_cid_3char is identic to labels_cid but the code is truncated to 3 characters (ICD-10 block)
labels_cid_3char = [x[:3] for x in labels_cid]

# labels_cid_aux is a list of the ICD-10 full codes present in the death certificate
labels_cid_aux = [ line.split('<>')[10].replace("'","") for line in texts ]
labels_cid_aux = [x[2:-2] for x in labels_cid_aux]
labels_cid_aux = [x.split(', ') for x in labels_cid_aux]

# Using sklearn package attribute an integer to each code that occures resulting in the variables:
# labels_int, labels_int_3char, labels_int_aux 
le3 = preprocessing.LabelEncoder()
le4 = preprocessing.LabelEncoder()
le4_aux = preprocessing.LabelEncoder()

char3 = le3.fit(labels_cid_3char)           
char4 = le4.fit(labels_cid)
char4_aux = le4_aux.fit([item for sublist in labels_cid_aux for item in sublist])

labels_int_3char = char3.transform(labels_cid_3char)
labels_int = char4.transform(labels_cid)

labels_int_aux = np.copy(labels_cid_aux)
for i in range(len(labels_int_aux)):
    labels_int_aux[i] = char4_aux.transform(labels_int_aux[i])

part_1a = [ line.split('<>')[1].lower() for line in texts ]
part_1b = [ line.split('<>')[2].lower() for line in texts ]
part_1c = [ line.split('<>')[3].lower() for line in texts ]
part_1d = [ line.split('<>')[4].lower() for line in texts ]
part_2 = [ line.split('<>')[5].lower() for line in texts ]
bic = [ line.split('<>')[6].lower() for line in texts ]
bic_admiss = [ line.split('<>')[7].lower() for line in texts ]
bic_sit = [ line.split('<>')[8].lower() for line in texts ]
ra = [ line.split('<>')[9].lower() for line in texts ]

tokenizer = Tokenizer(nb_words = max_features)
tokenizer.fit_on_texts(part_1a+part_1b+part_1c+part_1d+part_2+bic+bic_admiss+bic_sit+ra)

print('Loading death certificates...')
# attribute an integer to each token that occures in the texts 
# conversion of each dataset entry in a (7,200) shape matrix resulting in variables:

# data is a (5,200) matrix for the strings in death certificates
data = np.zeros((len(texts), maxsents, maxlen), dtype = 'int32')

death_cert = [part_1a,part_1b,part_1c,part_1d,part_2]
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
                        data[i,m,k] = tokenizer.word_index[word]
                        k = k + 1
                    
print('Loading bic...')

bic_components = [bic, bic_admiss, bic_sit]
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
                        data[i,5+m,k] = tokenizer.word_index[word]
                        k = k + 1

print('Loading autopsy reports...')

for i, sentences in enumerate(ra):
    sentences = tokenize.sent_tokenize( sentences )
    k = 0
    for j, sent in enumerate(sentences):
        if j < maxsents:
            wordTokens = text_to_word_sequence(sent)
            for _ , word in enumerate(wordTokens):
                if k < maxlen and tokenizer.word_index[word] < max_features:
                    data[i,8,k] = tokenizer.word_index[word]
                    k = k + 1

word_index = tokenizer.word_index

np.save('6_DICT.npy', word_index)
np.save('6_FULL_CODES.npy', le4)
np.save('6_BLOCKS.npy', le3)

#%%
print('Found %s unique tokens.' % len(word_index))

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
    labels_aux[i,:] = sum( to_categorical(labels_int_aux[i], nb_classes=num_classes))

#%%

print('Spliting the data into a training set and a validation set...')

X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train, X_test, y_train_aux, y_test_aux = train_test_split(data, labels_aux, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train, X_test, y_train_3char, y_test_3char = train_test_split(data, labels_3char, stratify = labels_cid, test_size = 0.25, random_state=42)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


print('Computing Initialization with Label Co-occurrence...')

train_labels_aux = [np.where(x != 0)[0] for x in y_train_aux]
train_labels_full = [np.where(x != 0)[0] for x in y_train]

train_labels_aux_cid = []
train_labels_full_cid = []

for i in range(len(train_labels_full)):
    train_labels_full_cid.extend(list(le4.inverse_transform(train_labels_full[i])))

for i in range(len(train_labels_aux)):
    train_labels_aux_cid.append(list(le4_aux.inverse_transform(train_labels_aux[i])))

extra_labels_cid = np.setdiff1d([item for sublist in labels_cid_aux for item in sublist],labels_cid)
extra_labels = le4_aux.transform(extra_labels_cid)

common_labels_cid = list(set([item for sublist in labels_cid_aux for item in sublist]).intersection(labels_cid))
common_labels = le4_aux.transform(common_labels_cid)

#%%

init_m_aux = np.zeros((num_classes,num_classes), dtype=np.float32)
bias_aux = np.zeros((num_classes,1), dtype=np.float32)

for i in range(len(train_labels_aux)):
    for j in range(len(train_labels_aux[i])):
        for k in range(len(train_labels_aux[i])):
            init_m_aux[train_labels_aux[i][j],train_labels_aux[i][k]] += 1

nmf = NMF(n_components=gru_output_size+embedding_dims)
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
nmf.fit(init_m_full)
init_m_full = nmf.components_
   
#%%
print('Build model...')

# Inputs
review_input = Input(shape=(maxsents,maxlen), dtype='int32')

# Embedding Layer
embedding_layer = Embedding(max_features, embedding_dims, 
                            input_length=maxlen)

# WORD-LEVEL
sentence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)

# Bidirectional GRU
l_gru = Bidirectional(GRU(gru_output_size, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(gru_output_size))(l_gru)
# Word-Level Attention Layer
l_att = AttLayer()(l_dense)
sentEncoder = Model(sentence_input, l_att)
review_encoder = TimeDistributed(sentEncoder)(review_input)

# SENTENCE_LEVEL
# Bidirectional GRU
l_gru_sent = Bidirectional(GRU(gru_output_size, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(gru_output_size))(l_gru_sent)
# Sentence-Level Attention Layer
postp = AttLayer()(l_dense_sent)

# Embedding Average
sentEmbed = Model(sentence_input, embedded_sequences)
review_fasttext = TimeDistributed(sentEmbed)(review_input)
fasttext = GlobalAveragePooling2D()(review_fasttext)

postp_aux = merge( [ postp , fasttext ] , mode='concat', concat_axis = 1 )

postp_aux_drop = Dropout(0.05)(postp_aux)

postp = Dense(output_dim=(gru_output_size+embedding_dims))(postp_aux_drop)

# Softmax
preds = Dense(y_train.shape[1], activation='softmax', weights=(init_m_full,np.zeros(y_train.shape[1])), name='full_code')(postp)
preds_3char = Dense(y_train_3char.shape[1], activation='softmax', name='block')(postp)
preds_aux = Dense(y_train_aux.shape[1], activation='sigmoid', weights=(init_m_aux,np.zeros(y_train_aux.shape[1])), name='aux')(postp)

model = Model(input = review_input, output = [preds, preds_3char, preds_aux])

model.compile(loss=['categorical_crossentropy','categorical_crossentropy','binary_crossentropy'], optimizer='adam', 
              metrics=['accuracy'], loss_weights = [0.8 , 0.85, 0.75])

#%%

model.fit(X_train, [y_train, y_train_3char, y_train_aux], batch_size=batch_size, nb_epoch=nb_epoch, 
          validation_data=(X_test, [y_test, y_test_3char, y_test_aux]), callbacks=[earlyStopping])

model.save('6_modelo_full_nmf.h5')

print('Predicting...')
[all_4, all_3, all_aux] = model.predict(X_test, batch_size=3)

print('Writing output...')

cid_pred = np.zeros([len(y_test),7], dtype = object)

for i in range(len(y_test)):
    top3_4 = np.argsort(all_4[i])[-3:]
    top3_3 = np.argsort(all_3[i])[-3:]
    cid_pred[i][0] = le4.inverse_transform(np.argmax(y_test[i]))
    for j in [1,2,3]:
        cid_pred[i][j] = le4.inverse_transform(top3_4[-j])
        cid_pred[i][3+j] = le3.inverse_transform(top3_3[-j])

np.savetxt('6_pred_full_nmf.txt', cid_pred, delimiter=" ", fmt="%s")
