#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np

np.random.seed(1337) # for reproducibility

from keras import backend as K
from keras.preprocessing.text import text_to_word_sequence
from keras.models import load_model
from keras.engine.topology import Layer
from keras import initializations
from nltk import tokenize
import jellyfish

# Custom Attention Layer
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
        self.att1 = weights        
        weighted_input = x*weights.dimshuffle(0,1,'x')          
        self.att2 = weighted_input        
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Set parameters:
max_features = 30000            # Maximum number of tokens in vocabulary
maxlen = 400                    # Maximum Length of each Sentence
maxsents = 9                    # Maximum Number of Sentences (5 for Death Certificate + 1 for Autopsy Report + 1 for Clinical Information Bulletin)

# Load dictionary
word_index = np.load('vocabulary.npy').item()
inv_word_index = {v: k for k, v in word_index.items()}

# Load ICD-10 to integer codes dictionary
le4 = np.load('full_codes.npy').item()
le3 = np.load('blocks.npy').item()

print('Load model...')
model = load_model('dnn_model.h5', custom_objects = {"AttLayer": AttLayer})

#%%

def PREDICT(part_1a, part_1b, part_1c, part_1d, part_2, bic, bic_admiss, bic_sit, ra):

    for i in range(len(part_1a)):
        if part_1a[i]!='': part_1a[i] = '873heGKe7I ' + part_1a[i].replace('.','') + ' 873heGKe7I'
        if part_1b[i]!='': part_1b[i] = '873heGKe7I ' + part_1b[i].replace('.','') + ' 873heGKe7I'
        if part_1d[i]!='': part_1d[i] = '873heGKe7I ' + part_1d[i].replace('.','') + ' 873heGKe7I'
        if part_1c[i]!='': part_1c[i] = '873heGKe7I ' + part_1c[i].replace('.','') + ' 873heGKe7I'
        if part_2[i]!='': part_2[i] = '873heGKe7I ' + part_2[i].replace('.','') + ' 873heGKe7I'
        if bic[i]!='': bic[i] =  bic[i].replace('.','')
        if bic_admiss[i]!='': bic_admiss[i] =  bic_admiss[i].replace('.','')
        if bic_sit[i]!='': bic_sit[i] =  bic_sit[i].replace('.','')
        if ra[i]!='': ra[i] = ra[i].replace('.','')
        
    texts = [part_1a, part_1b, part_1c, part_2, bic, bic_admiss, bic_sit, ra]

    data = np.zeros((len(texts[0]), maxsents, maxlen), dtype = 'int32')
    
# 'Loading Death certificates...'
    
    death_cert = [part_1a,part_1b,part_1c,part_1d,part_2]
        
    for m in range(len(death_cert)):
        part = death_cert[m]
        for i, sentences in enumerate(part):
            sentences = tokenize.sent_tokenize( sentences )
            k = 0
            for j, sent in enumerate(sentences):
                wordTokens = text_to_word_sequence(sent)
                for _ , word in enumerate(wordTokens):
                    # if the word is out-of-vocabulary it is substituted by the most
                    # similar word in the dictionary
                    if word_index.get(word) == None: 
                        aux = [(jellyfish.jaro_winkler(k,word),v) for k,v in word_index.items()]
                        if k < maxlen and max(aux)[1] < max_features:
                            data[i,m,k] = max(aux)[1]
                            k = k + 1
                    else:
                        if k < maxlen and word_index.get(word) < max_features:
                            data[i,m,k] = word_index.get(word)
                            k = k + 1
                        
# 'Loading bic...'
    
    bic_components = [bic, bic_admiss, bic_sit]
    for m in range(len(bic_components)):
        bic_part = bic_components[m]
        for i, sentences in enumerate(bic_part):
            sentences = tokenize.sent_tokenize( sentences )
            k = 0
            for j, sent in enumerate(sentences):
                wordTokens = text_to_word_sequence(sent)
                for _ , word in enumerate(wordTokens):
                    # if the word is out-of-vocabulary it is substituted by the most
                    # similar word in the dictionary
                    if word_index.get(word) == None: 
                        aux = [(jellyfish.jaro_winkler(k,word),v) for k,v in word_index.items()]
                        if k < maxlen and max(aux)[1] < max_features:
                            data[i,5+m,k] = max(aux)[1]
                            k = k + 1
                    else:
                        if k < maxlen and word_index.get(word) < max_features:
                            data[i,5+m,k] = word_index.get(word)
                            k = k + 1
    
# 'Loading autopsy reports...'
    
    for i, sentences in enumerate(ra):
        sentences = tokenize.sent_tokenize( sentences )
        k = 0
        for j, sent in enumerate(sentences):
            wordTokens = text_to_word_sequence(sent)
            for _ , word in enumerate(wordTokens):
                # if the word is out-of-vocabulary it is substituted by the most
                # similar word in the dictionary
                if word_index.get(word) == None: 
                    aux = [(jellyfish.jaro_winkler(k,word),v) for k,v in word_index.items()]
                    if k < maxlen and max(aux)[1] < max_features:
                        data[i,8,k] = max(aux)[1]
                        k = k + 1
                else:
                    if k < maxlen and word_index.get(word) < max_features:
                        data[i,8,k] = word_index.get(word)
                        k = k + 1
                            
# 'Predicting...'
    [all_4, all_3, aux] = model.predict(data)
        
    prediction_4 = []
    
    for i in range(len(texts[0])):
        top3_4 = np.argsort(all_4[i])[-1:]
        prediction_4.extend(le4.inverse_transform(top3_4))
            
    return prediction_4
