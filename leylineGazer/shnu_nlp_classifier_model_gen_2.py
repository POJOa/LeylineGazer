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
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import simplejson as json
import jieba
import logging
import sys
import pickle
from sklearn.preprocessing import minmax_scale
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
w2v_model = Word2Vec.load('./shnu/shnu_w2v_alt_trained_1500_uncut_lined.bin').wv

dest = "./shnu/classed_cut/"
files = os.listdir(dest)
sources = {}
file_map= {}
count=0
x_train_pos = []
x_train_pos_alt = []
text_train_pos = []
text_train_pos_alt = []

x_train_neg = []
x_train_neg_alt = []
text_train_neg = []
text_train_neg_alt = []

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





count = 1
all = len(pos)

for name in pos:
    print(count , ' / ' , all)
    count+=1
    f = open(dest+name+'.txt',errors='ignore')
    train_docs = f.readlines()
    for train_doc in train_docs:
        words = train_doc.split()
        vector = np.zeros(1500)
        word_num = 0
        for word in words:
            if word in w2v_model:
                vector += w2v_model[word]
                word_num += 1
        if word_num > 0:
            vector = vector / word_num
        x_train_pos.append(vector)


print('pos vector extracted')


count = 1
all = len(neg)
for name in neg:
    print(count , ' / ' , all)
    count+=1
    f = open(dest + name + '.txt',errors='ignore')
    train_docs = f.readlines()
    for train_doc in train_docs:
        words = train_doc.split()
        vector = np.zeros(1500)
        word_num = 0
        for word in words:
            if word in w2v_model:
                vector += w2v_model[word]
                word_num += 1
        if word_num > 0:
            vector = vector / word_num
        x_train_neg.append(vector)


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


print('svc linear')
svc_rbf=svm.SVC(kernel='linear')
svc_rbf.fit(train_x,train_y)
joblib.dump(svc_rbf,'./shnu/shnu_w2v_svc_linear_train_w2v_encore_uc_lined_1500.m')
preds = svc_rbf.predict(test_x)
print(metrics.classification_report(test_y, preds))
print(metrics.confusion_matrix(test_y, preds))

