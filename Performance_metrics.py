#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np

np.random.seed(1337) # for reproducibility

import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing 
from matplotlib import pyplot as plt
from sklearn import metrics

#%% LOAD PREDICTIONS

# The file predictions.txt has one array for each instance, organized in the following way:
# true label, 3 most probable 4 digit (full-codes) predicted labels, 3 most probable 3 digit (blocks) predicted labels

labels_pred = np.genfromtxt('example_predictions.txt', dtype = 'str')

#%%

# labels_cid has the true labels
labels_cid = [x[0] for x in labels_pred]
# labels_pred the predicted labels
labels_pred = [[x[1],x[2],x[3],x[4],x[5],x[6]] for x in labels_pred]

cid_4 = preprocessing.LabelEncoder()
cid_3 = preprocessing.LabelEncoder()
cid_1 = preprocessing.LabelEncoder()

# converting the 4 digit codes (full-codes) to integers
char_4 = cid_4.fit([x[:4] for x in labels_cid]+[x[:4] for x in [x[0] for x in labels_pred]])
# converting the 3 digit codes (blocks) to integers
char_3 = cid_3.fit([x[:3] for x in labels_cid]+[x[:3] for x in [x[0] for x in labels_pred]])
# converting the 1 digit codes (chapters) to integers
char_1 = cid_1.fit([x[:1] for x in labels_cid]+[x[:1] for x in [x[0] for x in labels_pred]])

# Integer values for the true labels
true_4 = char_4.transform([x[:4] for x in labels_cid])
true_3 = char_3.transform([x[:3] for x in labels_cid])
true_1 = char_1.transform([x[:1] for x in labels_cid])

# Integer values for the most probable predicted labels (full-code, block and chapter)
pred_4 = char_4.transform([x[:4] for x in [x[0] for x in labels_pred]])
pred_3 = char_3.transform([x[:3] for x in [x[0] for x in labels_pred]])
pred_1 = char_1.transform([x[:1] for x in [x[0] for x in labels_pred]])

#%% CLASS ACCURACY ANALYSIS

c_labels_cid = [x[:3] for x in labels_cid]

c_labels_pred = [x[:3] for x in [x[0] for x in labels_pred]]

for i in range(len(c_labels_cid)):
    if c_labels_cid[i] >= 'A00' and c_labels_cid[i] <= 'B99': 
        c_labels_cid[i] = 1 
    elif c_labels_cid[i] >= 'C00' and c_labels_cid[i] <= 'D48': 
        c_labels_cid[i] = 2
    elif c_labels_cid[i] >= 'D50' and c_labels_cid[i] <= 'D89': 
        c_labels_cid[i] = 3
    elif c_labels_cid[i] >= 'E00' and c_labels_cid[i] <= 'E90': 
        c_labels_cid[i] = 4
    elif c_labels_cid[i] >= 'F00' and c_labels_cid[i] <= 'F99': 
        c_labels_cid[i] = 5
    elif c_labels_cid[i] >= 'G00' and c_labels_cid[i] <= 'G99': 
        c_labels_cid[i] = 6
    elif c_labels_cid[i] >= 'H00' and c_labels_cid[i] <= 'H59': 
        c_labels_cid[i] = 7
    elif c_labels_cid[i] >= 'H60' and c_labels_cid[i] <= 'H95': 
        c_labels_cid[i] = 8
    elif c_labels_cid[i] >= 'I00' and c_labels_cid[i] <= 'I99': 
        c_labels_cid[i] = 9
    elif c_labels_cid[i] >= 'J00' and c_labels_cid[i] <= 'J99': 
        c_labels_cid[i] = 10
    elif c_labels_cid[i] >= 'K00' and c_labels_cid[i] <= 'K93': 
        c_labels_cid[i] = 11
    elif c_labels_cid[i] >= 'L00' and c_labels_cid[i] <= 'L99': 
        c_labels_cid[i] = 12
    elif c_labels_cid[i] >= 'M00' and c_labels_cid[i] <= 'M99': 
        c_labels_cid[i] = 13
    elif c_labels_cid[i] >= 'N00' and c_labels_cid[i] <= 'N99': 
        c_labels_cid[i] = 14
    elif c_labels_cid[i] >= 'O00' and c_labels_cid[i] <= 'O99': 
        c_labels_cid[i] = 15
    elif c_labels_cid[i] >= 'P00' and c_labels_cid[i] <= 'P96': 
        c_labels_cid[i] = 16
    elif c_labels_cid[i] >= 'Q00' and c_labels_cid[i] <= 'Q99': 
        c_labels_cid[i] = 17
    elif c_labels_cid[i] >= 'R00' and c_labels_cid[i] <= 'R99': 
        c_labels_cid[i] = 18
    elif c_labels_cid[i] >= 'S00' and c_labels_cid[i] <= 'T98': 
        c_labels_cid[i] = 19
    elif c_labels_cid[i] >= 'V01' and c_labels_cid[i] <= 'Y98': 
        c_labels_cid[i] = 20
    elif c_labels_cid[i] >= 'Z00' and c_labels_cid[i] <= 'Z99': 
        c_labels_cid[i] = 21
    else:
        c_labels_cid[i] = 22
                    
for i in range(len(c_labels_pred)):
    if c_labels_pred[i] >= 'A00' and c_labels_pred[i] <= 'B99': 
        c_labels_pred[i] = 1 
    elif c_labels_pred[i] >= 'C00' and c_labels_pred[i] <= 'D48': 
        c_labels_pred[i] = 2
    elif c_labels_pred[i] >= 'D50' and c_labels_pred[i] <= 'D89': 
        c_labels_pred[i] = 3
    elif c_labels_pred[i] >= 'E00' and c_labels_pred[i] <= 'E90': 
        c_labels_pred[i] = 4
    elif c_labels_pred[i] >= 'F00' and c_labels_pred[i] <= 'F99': 
        c_labels_pred[i] = 5
    elif c_labels_pred[i] >= 'G00' and c_labels_pred[i] <= 'G99': 
        c_labels_pred[i] = 6
    elif c_labels_pred[i] >= 'H00' and c_labels_pred[i] <= 'H59': 
        c_labels_pred[i] = 7
    elif c_labels_pred[i] >= 'H60' and c_labels_pred[i] <= 'H95': 
        c_labels_pred[i] = 8
    elif c_labels_pred[i] >= 'I00' and c_labels_pred[i] <= 'I99': 
        c_labels_pred[i] = 9
    elif c_labels_pred[i] >= 'J00' and c_labels_pred[i] <= 'J99': 
        c_labels_pred[i] = 10
    elif c_labels_pred[i] >= 'K00' and c_labels_pred[i] <= 'K93': 
        c_labels_pred[i] = 11
    elif c_labels_pred[i] >= 'L00' and c_labels_pred[i] <= 'L99': 
        c_labels_pred[i] = 12
    elif c_labels_pred[i] >= 'M00' and c_labels_pred[i] <= 'M99': 
        c_labels_pred[i] = 13
    elif c_labels_pred[i] >= 'N00' and c_labels_pred[i] <= 'N99': 
        c_labels_pred[i] = 14
    elif c_labels_pred[i] >= 'O00' and c_labels_pred[i] <= 'O99': 
        c_labels_pred[i] = 15
    elif c_labels_pred[i] >= 'P00' and c_labels_pred[i] <= 'P96': 
        c_labels_pred[i] = 16
    elif c_labels_pred[i] >= 'Q00' and c_labels_pred[i] <= 'Q99': 
        c_labels_pred[i] = 17
    elif c_labels_pred[i] >= 'R00' and c_labels_pred[i] <= 'R99': 
        c_labels_pred[i] = 18
    elif c_labels_pred[i] >= 'S00' and c_labels_pred[i] <= 'T98': 
        c_labels_pred[i] = 19
    elif c_labels_pred[i] >= 'V01' and c_labels_pred[i] <= 'Y98': 
        c_labels_pred[i] = 20
    elif c_labels_pred[i] >= 'Z00' and c_labels_pred[i] <= 'Z99': 
        c_labels_pred[i] = 21
    else:
        c_labels_pred[i] = 22
 
#%% MRR

lvl = 4

# FOR LVL = 4 (FULL-CODE) OR LVL = 3 (BLOCKS)
mrr = pd.DataFrame([x[:lvl] for x in labels_cid])
mrr['pred_1'] = [x[:lvl] for x in [x[0] for x in labels_pred]]
mrr['pred_2'] = [x[:lvl] for x in [x[1] for x in labels_pred]]
mrr['pred_3'] = [x[:lvl] for x in [x[2] for x in labels_pred]]

## FOR LVL = 1 (CHAPTERS)
#mrr = pd.DataFrame(c_labels_cid)
#mrr['pred_1'] = c_labels_pred
#c_labels_pred_2 = [x[:3] for x in [x[1] for x in labels_pred]]
#for i in range(len(c_labels_pred_2)):
#    if c_labels_pred_2[i] >= 'A00' and c_labels_pred_2[i] <= 'B99': 
#        c_labels_pred_2[i] = 1 
#    elif c_labels_pred_2[i] >= 'C00' and c_labels_pred_2[i] <= 'D48': 
#        c_labels_pred_2[i] = 2
#    elif c_labels_pred_2[i] >= 'D50' and c_labels_pred_2[i] <= 'D89': 
#        c_labels_pred_2[i] = 3
#    elif c_labels_pred_2[i] >= 'E00' and c_labels_pred_2[i] <= 'E90': 
#        c_labels_pred_2[i] = 4
#    elif c_labels_pred_2[i] >= 'F00' and c_labels_pred_2[i] <= 'F99': 
#        c_labels_pred_2[i] = 5
#    elif c_labels_pred_2[i] >= 'G00' and c_labels_pred_2[i] <= 'G99': 
#        c_labels_pred_2[i] = 6
#    elif c_labels_pred_2[i] >= 'H00' and c_labels_pred_2[i] <= 'H59': 
#        c_labels_pred_2[i] = 7
#    elif c_labels_pred_2[i] >= 'H60' and c_labels_pred_2[i] <= 'H95': 
#        c_labels_pred_2[i] = 8
#    elif c_labels_pred_2[i] >= 'I00' and c_labels_pred_2[i] <= 'I99': 
#        c_labels_pred_2[i] = 9
#    elif c_labels_pred_2[i] >= 'J00' and c_labels_pred_2[i] <= 'J99': 
#        c_labels_pred_2[i] = 10
#    elif c_labels_pred_2[i] >= 'K00' and c_labels_pred_2[i] <= 'K93': 
#        c_labels_pred_2[i] = 11
#    elif c_labels_pred_2[i] >= 'L00' and c_labels_pred_2[i] <= 'L99': 
#        c_labels_pred_2[i] = 12
#    elif c_labels_pred_2[i] >= 'M00' and c_labels_pred_2[i] <= 'M99': 
#        c_labels_pred_2[i] = 13
#    elif c_labels_pred_2[i] >= 'N00' and c_labels_pred_2[i] <= 'N99': 
#        c_labels_pred_2[i] = 14
#    elif c_labels_pred_2[i] >= 'O00' and c_labels_pred_2[i] <= 'O99': 
#        c_labels_pred_2[i] = 15
#    elif c_labels_pred_2[i] >= 'P00' and c_labels_pred_2[i] <= 'P96': 
#        c_labels_pred_2[i] = 16
#    elif c_labels_pred_2[i] >= 'Q00' and c_labels_pred_2[i] <= 'Q99': 
#        c_labels_pred_2[i] = 17
#    elif c_labels_pred_2[i] >= 'R00' and c_labels_pred_2[i] <= 'R99': 
#        c_labels_pred_2[i] = 18
#    elif c_labels_pred_2[i] >= 'S00' and c_labels_pred_2[i] <= 'T98': 
#        c_labels_pred_2[i] = 19
#    elif c_labels_pred_2[i] >= 'V01' and c_labels_pred_2[i] <= 'Y98': 
#        c_labels_pred_2[i] = 20
#    elif c_labels_pred_2[i] >= 'Z00' and c_labels_pred_2[i] <= 'Z99': 
#        c_labels_pred_2[i] = 21
#    else:
#        c_labels_pred_2[i] = 22       
#mrr['pred_2'] = c_labels_pred_2
#c_labels_pred_3 = [x[:3] for x in [x[2] for x in labels_pred]]
#for i in range(len(c_labels_pred_3)):
#    if c_labels_pred_3[i] >= 'A00' and c_labels_pred_3[i] <= 'B99': 
#        c_labels_pred_3[i] = 1 
#    elif c_labels_pred_3[i] >= 'C00' and c_labels_pred_3[i] <= 'D48': 
#        c_labels_pred_3[i] = 2
#    elif c_labels_pred_3[i] >= 'D50' and c_labels_pred_3[i] <= 'D89': 
#        c_labels_pred_3[i] = 3
#    elif c_labels_pred_3[i] >= 'E00' and c_labels_pred_3[i] <= 'E90': 
#        c_labels_pred_3[i] = 4
#    elif c_labels_pred_3[i] >= 'F00' and c_labels_pred_3[i] <= 'F99': 
#        c_labels_pred_3[i] = 5
#    elif c_labels_pred_3[i] >= 'G00' and c_labels_pred_3[i] <= 'G99': 
#        c_labels_pred_3[i] = 6
#    elif c_labels_pred_3[i] >= 'H00' and c_labels_pred_3[i] <= 'H59': 
#        c_labels_pred_3[i] = 7
#    elif c_labels_pred_3[i] >= 'H60' and c_labels_pred_3[i] <= 'H95': 
#        c_labels_pred_3[i] = 8
#    elif c_labels_pred_3[i] >= 'I00' and c_labels_pred_3[i] <= 'I99': 
#        c_labels_pred_3[i] = 9
#    elif c_labels_pred_3[i] >= 'J00' and c_labels_pred_3[i] <= 'J99': 
#        c_labels_pred_3[i] = 10
#    elif c_labels_pred_3[i] >= 'K00' and c_labels_pred_3[i] <= 'K93': 
#        c_labels_pred_3[i] = 11
#    elif c_labels_pred_3[i] >= 'L00' and c_labels_pred_3[i] <= 'L99': 
#        c_labels_pred_3[i] = 12
#    elif c_labels_pred_3[i] >= 'M00' and c_labels_pred_3[i] <= 'M99': 
#        c_labels_pred_3[i] = 13
#    elif c_labels_pred_3[i] >= 'N00' and c_labels_pred_3[i] <= 'N99': 
#        c_labels_pred_3[i] = 14
#    elif c_labels_pred_3[i] >= 'O00' and c_labels_pred_3[i] <= 'O99': 
#        c_labels_pred_3[i] = 15
#    elif c_labels_pred_3[i] >= 'P00' and c_labels_pred_3[i] <= 'P96': 
#        c_labels_pred_3[i] = 16
#    elif c_labels_pred_3[i] >= 'Q00' and c_labels_pred_3[i] <= 'Q99': 
#        c_labels_pred_3[i] = 17
#    elif c_labels_pred_3[i] >= 'R00' and c_labels_pred_3[i] <= 'R99': 
#        c_labels_pred_3[i] = 18
#    elif c_labels_pred_3[i] >= 'S00' and c_labels_pred_3[i] <= 'T98': 
#        c_labels_pred_3[i] = 19
#    elif c_labels_pred_3[i] >= 'V01' and c_labels_pred_3[i] <= 'Y98': 
#        c_labels_pred_3[i] = 20
#    elif c_labels_pred_3[i] >= 'Z00' and c_labels_pred_3[i] <= 'Z99': 
#        c_labels_pred_3[i] = 21
#    else:
#        c_labels_pred_3[i] = 22
#mrr['pred_3'] = c_labels_pred_3

mrr['pred_1'] = mrr[0] == mrr['pred_1']
mrr['pred_2'] = mrr[0] == mrr['pred_2']
mrr['pred_3'] = mrr[0] == mrr['pred_3']
mrr['k'] = [np.argmax( mrr[['pred_1','pred_2','pred_3']].iloc[x].get_values()) for x in range(len(mrr))]
mrr['k'] = np.where(mrr['k'] == 2, 1/3, mrr['k'])
mrr['k'] = np.where(mrr['k'] == 1, 1/2, mrr['k'])
mrr['k'] = np.where(mrr['k'] == 0, 1, mrr['k'])
mrr['bad'] = [sum(mrr[['pred_1','pred_2','pred_3']].iloc[x].get_values()) for x in range(len(mrr))]
mrr['k'] = np.where(mrr['bad'] == 0, 0, mrr['k'])
MRR = sum(mrr['k'])/len(mrr)
print(MRR)


#%% OCCURENCES IN CHAPTERS II, IX
# Accuracy and Macro-averaged Precision, Recall and F1-score for instances of Chapters II and IX

all_codes = pd.DataFrame(c_labels_cid)
all_codes['pred_int'] = c_labels_pred

class_ii = all_codes[all_codes[0]==2].index.tolist()
class_ix = all_codes[all_codes[0]==9].index.tolist()

true_4_ii = true_4[class_ii]
pred_4_ii = pred_4[class_ii]

true_3_ii = true_3[class_ii]
pred_3_ii = pred_3[class_ii]

true_4_ix = true_4[class_ix]
pred_4_ix = pred_4[class_ix]

true_3_ix = true_3[class_ix]
pred_3_ix = pred_3[class_ix]

print('\n -> OVERALL ACCURACY full-code (II): %s' % metrics.accuracy_score(true_4_ii,pred_4_ii))
print('\n      -> Precision (II): %s' % metrics.precision_score(true_4_ii,pred_4_ii,average='macro'))
print('\n      -> Recall (II): %s' % metrics.recall_score(true_4_ii,pred_4_ii,average='macro'))
print('\n      -> F1 (II): %s' % metrics.f1_score(true_4_ii,pred_4_ii,average='macro'))
print('\n -> OVERALL ACCURACY full-code (IX): %s' % metrics.accuracy_score(true_4_ix,pred_4_ix))
print('\n      -> Precision (IX): %s' % metrics.precision_score(true_4_ix,pred_4_ix,average='macro'))
print('\n      -> Recall (IX): %s' % metrics.recall_score(true_4_ix,pred_4_ix,average='macro'))
print('\n      -> F1 (IX): %s' % metrics.f1_score(true_4_ix,pred_4_ix,average='macro'))

print('\n -> OVERALL ACCURACY block (II): %s' % metrics.accuracy_score(true_3_ii,pred_3_ii))
print('\n      -> Precision (II): %s' % metrics.precision_score(true_3_ii,pred_3_ii,average='macro'))
print('\n      -> Recall (II): %s' % metrics.recall_score(true_3_ii,pred_3_ii,average='macro'))
print('\n      -> F1 (II): %s' % metrics.f1_score(true_3_ii,pred_3_ii,average='macro'))
print('\n -> OVERALL ACCURACY block (IX): %s' % metrics.accuracy_score(true_3_ix,pred_3_ix))
print('\n      -> Precision (IX): %s' % metrics.precision_score(true_3_ix,pred_3_ix,average='macro'))
print('\n      -> Recall (IX): %s' % metrics.recall_score(true_3_ix,pred_3_ix,average='macro'))
print('\n      -> F1 (IX): %s' % metrics.f1_score(true_3_ix,pred_3_ix,average='macro'))


#%% PRECISION, RECALL, F1-SCORE FOR SPECIFIC CODE

code = 'I20'
custom = pd.DataFrame([x[:len(code)] for x in labels_cid])
custom['pred'] = [x[:len(code)] for x in [x[0] for x in labels_pred]]
custom['code_occur'] = (custom[0]==code) | (custom['pred']==code)
custom['correct'] = custom[0]==custom['pred']
custom = custom[ custom['code_occur'] | custom['correct'] ]
custom = custom.drop(['code_occur', 'correct'],1)
custom[0] = custom[0] == code
custom['pred'] = custom['pred'] == code
print('%s cases of %s' %(np.sum(custom[0]),code))
print(metrics.confusion_matrix(custom[0],custom['pred']))
print('Accuracy: %s' % metrics.accuracy_score(custom[0], custom['pred']))
print('Precision: %s' % metrics.precision_score(custom[0],custom['pred']))
print('Recall: %s' % metrics.recall_score(custom[0],custom['pred']))
print('F1-Score: %s' % metrics.f1_score(custom[0],custom['pred']))

#%% TOP 20 CANCERS
# Precision Recall and F1-Score for the 20 most common cancer types in the dataset

import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 13}

matplotlib.rc('font', **font)

cancer_ = ['C34', 'C18','C50', 'C80', 'C25', 'C43', 'C85','C16','C15','C71','C22','C67','C64','C92', 'C56','C26','C20','C90', 'C19', 'C61']

cancer = pd.DataFrame(cancer_)
cancer['occurr'] = 0

for i in range(len(cancer)):
    code = cancer[0][i]
    custom = pd.DataFrame([x[:len(code)] for x in labels_cid])
    custom['pred'] = [x[:len(code)] for x in [x[0] for x in labels_pred]]
    custom['code_occur'] = (custom[0]==code) | (custom['pred']==code)
    custom['correct'] = custom[0]==custom['pred']
    custom = custom[ custom['code_occur'] | custom['correct'] ]
    custom = custom.drop(['code_occur', 'correct'],1)
    custom[0] = custom[0] == code
    custom['pred'] = custom['pred'] == code
    cancer['occurr'][i] = np.sum(custom[0])
    cancer.set_value(i,'p', metrics.precision_score(custom[0],custom['pred']))
    cancer.set_value(i,'r', metrics.recall_score(custom[0],custom['pred']))
    cancer.set_value(i,'f', metrics.f1_score(custom[0],custom['pred']))
   
cancer = cancer.sort('occurr', ascending=False)    
cancer = cancer.reset_index(drop=True)

plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(15,7.5))

ax = fig.add_subplot(111)
ax2 = ax.twinx()

plt.ylim([0.6,1])

plt.grid(b=True, which='major', linestyle='--', alpha=0.4)

cancer.occurr.plot(kind='bar', color='grey', ax=ax, align='center')

plt.plot(cancer.p,'ro', label = 'Precision')
plt.plot(cancer.r, 'b.', label = 'Recall')
plt.plot(cancer.f, 'g^', label = 'F1-Score')

plt.xticks(cancer.index,cancer_, rotation=45)
ax.set_ylabel('Occurrences in test data')
plt.title('Performance metrics for the 10 most common ICD-10 codes in dataset')

plt.legend()
 
plt.show()
#fig.savefig('top10_performance', dpi=1000,  bbox_inches='tight')

#%% DATASET ANALYSIS
# How many occurences of each chapter in the dataset

classes = np.zeros(22)

classes[0] = len([x for x in labels_cid if x < 'C000'])

classes[1] = len([x for x in [x for x in labels_cid if x > 'C0'] if x < 'D49'])

classes[2] = len([x for x in [x for x in labels_cid if x > 'D5'] if x < 'D90'])

classes[3] = len([x for x in [x for x in labels_cid if x > 'E0'] if x < 'E91'])

classes[4] = len([x for x in [x for x in labels_cid if x > 'F0'] if x < 'G0'])

classes[5] = len([x for x in [x for x in labels_cid if x > 'G0'] if x < 'H0'])

classes[6] = len([x for x in [x for x in labels_cid if x > 'H0'] if x < 'H60'])

classes[7] = len([x for x in [x for x in labels_cid if x > 'H6'] if x < 'H96'])

classes[8] = len([x for x in [x for x in labels_cid if x > 'I0'] if x < 'J00'])

classes[9] = len([x for x in [x for x in labels_cid if x > 'J0'] if x < 'K00'])

classes[10] =len([x for x in [x for x in labels_cid if x > 'K0'] if x < 'K94'])

classes[11] =len([x for x in [x for x in labels_cid if x > 'L0'] if x < 'M00'])

classes[12] =len([x for x in [x for x in labels_cid if x > 'M0'] if x < 'N00'])

classes[13] =len([x for x in [x for x in labels_cid if x > 'N0'] if x < 'O00'])

classes[14] =len([x for x in [x for x in labels_cid if x > 'O0'] if x < 'P00'])

classes[15] =len([x for x in [x for x in labels_cid if x > 'P0'] if x < 'P97'])

classes[16] =len([x for x in [x for x in labels_cid if x > 'Q0'] if x < 'R00'])

classes[17] =len([x for x in [x for x in labels_cid if x > 'R0'] if x < 'S00'])

classes[18] =len([x for x in [x for x in labels_cid if x > 'S0'] if x < 'T99'])

classes[19] = len([x for x in [x for x in labels_cid if x > 'V01'] if x < 'Y99'])

classes[20] = len([x for x in labels_cid if x > 'Z0'])

classes[21] = len([x for x in [x for x in labels_cid if x > 'U0'] if x < 'V00'])

for i in range(len(classes)):
    print('\nClass %s: %s (%.3f)' % ((i+1), classes[i], 100*classes[i]/len(labels_cid)))

#%% ACCURACY, MACRO-AVERAGED PRECISION RECALL AND F1-SCORE FOR FULL-CODES, BLOCK AND CHAPTER

p_per_class = metrics.precision_score(c_labels_cid,c_labels_pred,average=None,labels=list(set(c_labels_cid)))
r_per_class = metrics.recall_score(c_labels_cid,c_labels_pred,average=None,labels=list(set(c_labels_cid)))
f1_per_class = metrics.f1_score(c_labels_cid,c_labels_pred,average=None,labels=list(set(c_labels_cid)))

print('\n -> OVERALL ACCURACY (FULL-CODE): %s' % metrics.accuracy_score(true_4,pred_4))
print('\n      -> Precision (FULL-CODE): %s' % metrics.precision_score(true_4,pred_4,average='macro'))
print('\n      -> Recall (FULL-CODE): %s' % metrics.recall_score(true_4,pred_4,average='macro'))
print('\n      -> F1 (FULL-CODE): %s' % metrics.f1_score(true_4,pred_4,average='macro'))
print('\n -> OVERALL ACCURACY (BLOCKS): %s' % metrics.accuracy_score(true_3,pred_3))
print('\n      -> Precision (BLOCKS): %s' % metrics.precision_score(true_3,pred_3,average='macro'))
print('\n      -> Recall (BLOCKS): %s' % metrics.recall_score(true_3,pred_3,average='macro'))
print('\n      -> F1 (BLOCKS): %s' % metrics.f1_score(true_3,pred_3,average='macro'))
print('\n -> OVERALL ACCURACY (CHAPTER): %s' % metrics.accuracy_score(c_labels_cid,c_labels_pred))
print('\n      -> Precision (CHAPTER): %s' % metrics.precision_score(c_labels_cid,c_labels_pred,average='macro'))
print('\n      -> Recall (CHAPTER): %s' % metrics.recall_score(c_labels_cid,c_labels_pred,average='macro'))
print('\n      -> F1 (CHAPTER): %s' % metrics.f1_score(c_labels_cid,c_labels_pred,average='macro'))
print('\n -> PRECISION; RECALL; F1 SCORE: ')
for i in range(len(f1_per_class)):
    print('\n   | CLASS %s: %s  ;  %s  ;  %s' % (list(set(c_labels_cid))[i], p_per_class[i], r_per_class[i], f1_per_class[i]))
