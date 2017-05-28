from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import numpy
import os
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

import jieba
import logging
import sys

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line,errors='ignore').split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line,errors='ignore').split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

dest = "/Users/bytenoob/PycharmProjects/leylineGazer/leylineGazer/shnu/classed_cut/"
files = os.listdir(dest)
sources = {}
file_map= {}
count=0
for name in files:
    sources[dest+name] = 'TRAIN_'+name
    file_map[name]=count


#sentences = LabeledLineSentence(sources)
#model = Doc2Vec(min_count=10, window=5, size=100, sample=1e-4, negative=5, workers=4)
#s = sentences.to_array()
#model.build_vocab(s)
#model.train(sentences.sentences_perm(),total_examples=len(s), epochs=15)
#model.save('./shnu/shnu_d2v_trained.bin')
model = Doc2Vec.load('./shnu/shnu_d2v_trained.bin')
tags = model.docvecs.doctags.items()
train_arrays=numpy.zeros((len(tags),100))
train_labels = numpy.zeros(len(tags))
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
'res.ec',
'fineart',
'xlzx',
'waiyu',
'fazheng',
'ictt',

     ]
for (k,v) in tags:
    tag_parts = k.split('_')
    try:
        indice = int(tag_parts[2])
    except:
        continue
    name = tag_parts[1]
    if name.replace(".txt","") in pos:
        train_labels[indice] = 1
    elif name.replace(".txt","") in neg:
        train_labels[indice] = 0
    else:
        continue
    train_arrays[indice] = model.docvecs[k]
    #train_labels[indice] = file_map[name]

test_arrays = []
true_labels=[]

a=open("/Users/bytenoob/PycharmProjects/leylineGazer/leylineGazer/shnu/classed_cut/xxjd.txt")
b=open("/Users/bytenoob/PycharmProjects/leylineGazer/leylineGazer/shnu/classed_cut/renwen.txt")
test_content1=a.readlines()
test_content2=b.readlines()
for i in test_content1:
    v = model.infer_vector(i.split())
    test_arrays.append(v)
    true_labels.append(1)
for i in test_content2:
    v = model.infer_vector(i.split())
    test_arrays.append(v)
    true_labels.append(0)
#print('classifier init')
#classifier = LogisticRegression(class_weight={0:0.38,1:0.62})
#print('classifier fit')
#classifier.fit(train_arrays, train_labels)
#RF = RandomForestClassifier(n_estimators=1200,max_depth=14,class_weight={0:0.3,1:0.7})
#RF.fit(train_arrays, train_labels)
print('classifier init')
GBDT = GradientBoostingClassifier(n_estimators=1000,max_depth=14)
GBDT.fit(train_arrays, train_labels)
joblib.dump(GBDT, "./shnu/train_model_gbdt.m")
print('rf fit')
RF = RandomForestClassifier(n_estimators=1200,max_depth=14,class_weight={0:0.3,1:0.7})
RF.fit(train_arrays, train_labels)
joblib.dump(RF, "./shnu/train_model_rf.m")
print('classifier fit')
classifier = LogisticRegression(class_weight={0:0.38,1:0.62})
classifier.fit(train_arrays, train_labels)
joblib.dump(classifier, "./shnu/train_model_rf.m")

test_labels_LR=[]
test_labels_RF=[]
test_labels_GBDT=[]
count = 0
for i in range(len(test_arrays)):
     print(count , ' / ', len(test_arrays))
     count+=1
     test_labels_LR.append(classifier.predict([test_arrays[i]]))
     test_labels_RF.append(RF.predict([test_arrays[i]]))
     test_labels_GBDT.append(GBDT.predict([test_arrays[i]]))
print("LR:")
print(classifier.score(test_labels_LR, true_labels))
print(confusion_matrix(test_labels_LR,true_labels))
print("RF:")
print(metrics.accuracy_score(test_labels_RF,true_labels))
print(confusion_matrix(test_labels_RF,true_labels))
print("GBDT:")
print(metrics.accuracy_score(test_labels_GBDT,true_labels))
print(confusion_matrix(test_labels_GBDT,true_labels))

'''
train_arrays = numpy.zeros((18293, 100))
train_labels = numpy.zeros(18293)
test_arrays = []
true_labels=[]
train_data=[]
train_lb=[]
for i in range(18293):
    if(i<=12988):
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays[i] = model.docvecs[prefix_train_neg]
        train_labels[i] = 0
    if(i>12988 and i<=18292):
        j=i-12989
        prefix_train_pos = 'TRAIN_POS_' + str(j)
        train_arrays[i]=model.docvecs[prefix_train_pos]
        train_labels[i]=1


a=open("/Volumes/Macintosh HD/Users/RayChou/Downloads/情感分析训练语料/pos_test.txt")
b=open("/Volumes/Macintosh HD/Users/RayChou/Downloads/情感分析训练语料/neg_test.txt")
test_content1=a.readlines()
test_content2=b.readlines()
for i in test_content1:
    test_arrays.append(model.infer_vector(i))
    true_labels.append(1)
for i in test_content2:
    test_arrays.append(model.infer_vector(i))
    true_labels.append(0)
classifier = LogisticRegression(class_weight={0:0.38,1:0.62})
classifier.fit(train_arrays, train_labels)
RF = RandomForestClassifier(n_estimators=1200,max_depth=14,class_weight={0:0.3,1:0.7})
RF.fit(train_arrays, train_labels)
GBDT = GradientBoostingClassifier(n_estimators=1000,max_depth=14)
GBDT.fit(train_arrays, train_labels)
test_labels_LR=[]
test_labels_RF=[]
test_labels_GBDT=[]
for i in range(len(test_arrays)):
    test_labels_LR.append(classifier.predict(test_arrays[i]))
    test_labels_RF.append(RF.predict(test_arrays[i]))
    test_labels_GBDT.append(GBDT.predict(test_arrays[i]))
print("LR:")
print(classifier.score(test_labels_LR, true_labels))
print(confusion_matrix(test_labels_LR,true_labels))
print("RF:")
print(metrics.accuracy_score(test_labels_RF,true_labels))
print(confusion_matrix(test_labels_RF,true_labels))
print("GBDT:")
print(metrics.accuracy_score(test_labels_GBDT,true_labels))
print(confusion_matrix(test_labels_GBDT,true_labels))
'''