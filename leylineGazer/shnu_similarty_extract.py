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
client = MongoClient('mongodb://localhost:27017/')
db = client.shnu_site
news_collection = db.News
world = news_collection.find().sort("_id",1)
corpora_documents = []
raw_documents = []
'''
sum = world.count()
count = 1
for entity in world:
    print(str(count) +' / '+str(sum))
    count+=1
    raw_documents.append(entity['text'])
    item_str = jieba.cut(entity['text'])
    words_of_doc = []
    for i in item_str:
        words_of_doc.append(i)
    corpora_documents.append(words_of_doc)
try:
    print('dump json')
    txt = json.dumps(corpora_documents)
    print('cut save')
    with open('./shnu/similarity_cut.txt','wb') as fW:
        for i in corpora_documents:
            for ii in i:
                t=(ii + '|').encode()
                fW.write(t)
            fW.write('\n'.encode())
except Exception as e:
    print('err')
    print(str(e))

print('dict')
'''
'''
dictionary = corpora.Dictionary(corpora_documents)
print('dict save')

count = 0
with open('./shnu/similarity_cut.txt') as fW:
    for i in fW:
        count += 1
        print(count)
        words_of_docs = []
        for ii in i.split('|'):
            words_of_docs.append(ii)
        corpora_documents.append(words_of_docs)
dictionary=corpora.Dictionary.load('./shnu/similarity_dict.txt')
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
'''
similarity= similarities.Similarity.load('./shnu/similarity.bin')
test_data_1 = world[0]['text']
test_cut_raw_1 = jieba.cut(test_data_1)
#dictonary = corpora.Dictionary([test_cut_raw_1])
dictionary=corpora.Dictionary.load('./shnu/similarity_dict.txt')
test_corpus_1 = dictionary.doc2bow(test_cut_raw_1)
similarity.num_best = 10
print(similarity[test_corpus_1])  # 返回最相似的样本材料,(index_of_document, similarity) tuples

