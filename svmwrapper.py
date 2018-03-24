#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import sklearn
import pickle
import numpy as np
from keras.models import Model
from keras.utils.np_utils import to_categorical
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Wraper over the SVM classifier from sklearn
class SVMWrapper(Model):

    def __init__(self, C=1.0, use_idf=True, two_vectors=True, filename=None, **kwargs):
        self.svmmodel = LinearSVC( C=C , random_state=0 )
        self.vect1 = TfidfVectorizer(norm=None, use_idf=use_idf, min_df=0.0)
        self.vect2 = TfidfVectorizer(norm=None, use_idf=use_idf, min_df=0.0)
        self.output_dim = 0
        self.two_vectors = two_vectors
        if filename is not None: self.load(filename)

    def build_representation( self, x, fit=False ):
        if not(self.two_vectors):
            auxX = [ ' \n '.join( [ ' '.join( [ 'w_'+str(token) for token in field if token != 0 ] ) for field in instance ] ) for instance in x ]
            if fit : return self.vect1.fit_transform(auxX).todense()
            else : return self.vect1.transform(auxX).todense()
        auxX1 = [ ' \n '.join( [ ' '.join( [ 'w_'+str(token) for token in field if token != 0 ] ) for field in instance[0:5,] ] ) for instance in x ]
        auxX1 = [ 'main \n ' + x for x in auxX1 ]
        if fit : auxX1 = self.vect1.fit_transform(auxX1)
        else : auxX1 = self.vect1.transform(auxX1)
        auxX2 = [ ' \n '.join( [ ' '.join( [ 'w_'+str(token) for token in field if token != 0 ] ) for field in instance[6:9,] ] ) for instance in x ]
        auxX2 = [ 'aux \n ' + x for x in auxX2 ]
        if fit : auxX2 = self.vect2.fit_transform(auxX2)
        else : auxX2 = self.vect2.transform(auxX2)
        return np.concatenate( [auxX1.todense() , auxX2.todense()] , axis=1)
        
    def fit( self, x, y, validation_data=None):
        auxX = self.build_representation(x,fit=True)
        auxY = y
        self.svmmodel.fit( auxX , np.array([ np.argmax(i) for i in auxY ]) )
        self.output_dim = auxY.shape[1]
        if validation_data is None: return None
        res = self.evaluate( validation_data[0] , validation_data[1] )
        print("Accuracy in validation data =",res)
        return None
		
    def predict(self, x):
        auxX = self.build_representation(x,fit=False)
        auxY = self.svmmodel.predict(auxX)
        auxY = to_categorical(auxY)
        if auxY.shape[1] < self.output_dim:
            npad = ((0, 0), (0, self.output_dim-auxY.shape[1]))
            auxY = np.pad(auxY, pad_width=npad, mode='constant', constant_values=0)
        return [ auxY, [], [] ]
        
    def evaluate(self, x, y):
        auxX = self.build_representation(x,fit=False)
        auxY = y
        auxY = np.array([ np.argmax(i) for i in auxY ])
        return sklearn.metrics.accuracy_score(y_true=auxY,y_pred=self.svmmodel.predict(auxX))
	
    def save(self, filename):
        f = open(filename, "wb")
        pickle.dump(self.svmmodel, f)
        pickle.dump(self.vect1, f)
        pickle.dump(self.vect2, f)
        pickle.dump(self.output_dim, f)
        pickle.dump(self.two_vectors, f)
        f.close()
	
    def load(self, filename): 
        f = open(filename, "rb")
        self.svmmodel = pickle.load(f)
        self.vect1 = pickle.load(f)
        self.vect2 = pickle.load(f)
        self.output_dim = pickle.load(f)
        self.two_vectors = pickle.load(f)
        f.close()
