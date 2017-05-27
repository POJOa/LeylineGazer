import os
import sys

import logging
from gensim import similarities
import pymongo
from gensim import corpora
import jieba
from pymongo import MongoClient
import simplejson as json
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
world = open('./shnu/shnu_cut.txt')
corpora_documents = []
raw_documents = []

sum = world
for entity in world:
    corpora_documents.append(entity.split(" "))

print('dict')
'''
'''
dictionary = corpora.Dictionary(corpora_documents)
print('dict save')
corpora.Dictionary.save(dictionary,'./shnu/similarity_dict.txt')
print('corpus')
count = 0
sum = len(corpora_documents)
corpus = []
for text in corpora_documents:
    count+=1
    print(str(count) + ' / '+str(sum))
    corpus.append(dictionary.doc2bow(text))

#corpus = [dictionary.doc2bow(text) for text in corpora_documents]
print('similarity')

similarity = similarities.Similarity('similarity-idx',corpus,num_features=564517,chunksize=10000)
print('similarity save')

similarities.Similarity.save(similarity,'./shnu/similarity.bin')