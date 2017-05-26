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
import jieba

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
                    yield TaggedDocument(jieba.cut(utils.to_unicode(line)), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(jieba.cut(utils.to_unicode(line)), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

dest = "/Users/bytenoob/PycharmProjects/leylineGazer/leylineGazer/shnu/classed/"
files = os.listdir(dest)
sources = {}
for name in files:
    sources[dest+files] = 'TRAIN_'+name


sentences = LabeledLineSentence(sources)
model = Doc2Vec(min_count=10, window=15, size=100, sample=1e-4, negative=10, workers=4)
model.save('./d2v.bin')
model.build_vocab(sentences.to_array())
for epoch in range(10):
    model.train(sentences.sentences_perm())

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