from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import numpy as np
import os
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split

import simplejson as json
import jieba
import logging
import sys
import pickle

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
#w2v_model = Word2Vec.load('./shnu/shnu_w2v_alt.bin').wv
w2v_model = KeyedVectors.load_word2vec_format('./juice.bin',unicode_errors='ignore',binary=True)

dest = "./shnu/classed_cut/"
files = os.listdir(dest)
sources = {}
file_map= {}
count=0
x_train_pos = []
x_train_neg = []
x_test = []

pos=['fb',
'shenghuan',
'xxjd',
'mathsc',
'kjc',
'bc',
'jrxy',
'xxb',
'jiangong',
'hxzx',
'res.chem']

neg=['xzx',
'renwen',
'xiejin',
'shcas',
'marx',
'iccs',
'cice',
'shkch',
'zhexue',
'jjc',
'fineart',
'xlzx',
'waiyu',
'fazheng',
'ictt',

     ]
'''
count = 1
all = len(pos)
for name in pos:
    print(count , ' / ' , all)
    count+=1
    f = open(dest+name+'.txt')
    train_docs = f.readlines()
    for train_doc in train_docs:
        words = train_doc.split()
        vector = np.zeros(400)
        word_num = 0
        for word in words:
            if word in w2v_model:
                vector += w2v_model[word]
                word_num += 1
        if word_num > 0:
            vector = vector / word_num
        x_train_pos.append(vector)
#f=open('./shnu/x_train_pos.v','wb')
'''
with open('./shnu/x_train_pos.v','rb') as f:
    x_train_pos = pickle.load(f)
print('pos vector extracted')

'''
count = 1
all = len(neg)
for name in neg:
    print(count , ' / ' , all)
    count+=1
    f = open(dest + name + '.txt')
    train_docs = f.readlines()
    for train_doc in train_docs:
        words = train_doc.split()
        vector = np.zeros(400)
        word_num = 0
        for word in words:
            if word in w2v_model:
                vector += w2v_model[word]
                word_num += 1
        if word_num > 0:
            vector = vector / word_num
        x_train_neg.append(vector)
pickle.dump(x_train_pos,f)
'''
with open('./shnu/x_train_neg.v','rb') as f:
    x_train_neg = pickle.load(f)
print('neg vector extracted')

train=[]
train.extend(x_train_pos)
train.extend(x_train_neg)

target=[1]*len(x_train_pos)+[0]*len(x_train_neg)
train_x,test_x, train_y, test_y = train_test_split(train,
                                                   target,
                                                   test_size = 0.2,
                                                   random_state = 0)

print(len(train_x), len(test_x))
print(len(train_y), len(test_y))

print('random ok')

svclf = svm.SVC(kernel = 'rbf')
print('fitting')

svclf.fit(train_x,train_y)
joblib.dump(svclf,'./shnu/shnu_w2v_svc_rbf.m')
print('1st dumped')

preds = svclf.predict(test_x)
num = 0
preds = preds.tolist()
for i,pred in enumerate(preds):
    if int(pred) == int(test_y[i]):
        num += 1
print('precision_score:' + str(float(num) / len(preds)))

svclf = svm.SVC(kernel = 'linear')
print('fitting')

svclf.fit(train_x,train_y)
joblib.dump(svclf,'./shnu/shnu_w2v_svc_linear.m')
print('2nd dumped')

preds = svclf.predict(test_x)
num = 0
preds = preds.tolist()
for i,pred in enumerate(preds):
    if int(pred) == int(test_y[i]):
        num += 1
print('precision_score:' + str(float(num) / len(preds)))
