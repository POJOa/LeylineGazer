from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import numpy as np
import os
from random import shuffle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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

from sklearn.preprocessing import minmax_scale

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
#w2v_model = Word2Vec.load('./shnu/shnu_w2v_alt.bin').wv
jieba.enable_parallel(4)
dest = "./shnu/"
files = os.listdir(dest)
sources = {}
file_map= {}
count=0
x_train_text_pos=[]
x_train_text_neg=[]

train_text_x =[]
test_text_x = []
train=[]

pos=['classification_pos']

neg=['classification_neg']

with open('./shnu/shnu_train_raw.pkl','rb') as f:
    train_raw = pickle.load(f,errors='ignore')

for name in pos:
    f = open(dest+name+'.json',errors='ignore')
    train_docs = json.load(f)
    all = len(train_docs)
    count=0

    for train_doc in train_docs:
        words = jieba.cut(train_doc['text'])
        x_train_text_pos.append(' '.join(words))
        print(count,'/',all)
        count+=1
print('pos vector extracted')


for name in neg:

    f = open(dest+name+'.json',errors='ignore')
    train_docs = json.load(f)
    all = len(train_docs)
    count=0
    for train_doc in train_docs:
        words = jieba.cut(train_doc['text'])
        x_train_text_neg.append(' '.join(words))
        print(count,'/',all)
        count+=1
print('neg vector extracted')


x_train_text=[]
x_train_text.extend(x_train_text_pos)
x_train_text.extend(x_train_text_neg)
y_train_text=[1]*len(x_train_text_pos)+[0]*len(x_train_text_neg)

count_v0= CountVectorizer()
counts_all = count_v0.fit_transform(train_raw)
count_v1= CountVectorizer(vocabulary=count_v0.vocabulary_)
counts_train = count_v1.fit_transform(x_train_text)
tfidftransformer = TfidfTransformer()
x_train_text = tfidftransformer.fit(counts_train).transform(counts_train)



m=joblib.load('./shnu/shnu_svm_l.m')
print('svm linear tf-idf')
preds = m.predict(x_train_text)
print(metrics.classification_report(y_train_text, preds))
print(metrics.confusion_matrix(y_train_text, preds))


m=joblib.load('./shnu/shnu_nb_b.m')
print('BernoulliNB')
preds = m.predict(x_train_text)
print(metrics.classification_report(y_train_text, preds))
print(metrics.confusion_matrix(y_train_text, preds))

m=joblib.load('./shnu/shnu_nb_m.m')
print('MultinomialNB')
preds = m.predict(x_train_text)
print(metrics.classification_report(y_train_text, preds))
print(metrics.confusion_matrix(y_train_text, preds))

