import os

import sys
import pickle

import logging
from multiprocessing.pool import Pool
import multiprocessing
from gensim import similarities
import pymongo
from gensim import corpora
import jieba
from pymongo import MongoClient
from bson.json_util import dumps

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format=' %(process)d: %(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
manager = multiprocessing.Manager()

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



all_deleted = []

def tsk(i,sim,idmap):
    client = MongoClient('mongodb://localhost:27017/')
    db = client.shnu_site
    news_collection = db.News
    deleted = []
    print('process start')
    self_id = i['_id']
    if self_id in all_deleted:
        print(str(self_id) + ' has already been deleted')
        return

    for (k,v) in sim:
        target_id=idmap[k]
        if v>0.9999 and target_id!=self_id:
            try:
                print(str(target_id) + ' will be removed due to high similarity with  '+str(self_id))
                news_collection.remove({"_id":target_id})
                deleted.append(target_id)
            except Exception as e:
                print(str(e) + ' is failed')
    print('process end')

    all_deleted.append(l for l in deleted)


# noinspection PyArgumentList

def main():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.shnu_site
    news_collection = db.News
    news_collection_raw = db.News_backup
    world = news_collection.find(no_cursor_timeout=True).sort("_id", 1)
    world_raw = news_collection_raw.find(no_cursor_timeout=True).sort("_id", 1)
    jieba.enable_parallel(4)


    idmap = []

    for i in world_raw:
        idmap.append(i['_id'])

    similarity = similarities.Similarity.load('./shnu/similarity.bin')
    similarity.num_best = 5
    dictionary = corpora.Dictionary.load('./shnu/similarity_dict.txt')

    count = 0
    resColle = []
    print('cutting text')
    cut_texts = [jieba.cut(i['text']) for i in world]
    print('generating corpus')
    corpus = [dictionary.doc2bow(text) for text in cut_texts]
    print('getting sim results')
    simResColle = [(i,sim) for sim in similarity[corpus]]

    with open('pk.pkl', 'wb') as f:  # open file with write-mode
        picklestring = pickle.dump(simResColle,f)  # serialize object

    count = 0
    all = len(simResColle)
    for (i,sim) in simResColle:
        count += 1
        print('processing ',str(count),' / ',str(all))
        tsk(i,sim,idmap)

    print(all_deleted)

if __name__ == '__main__':
    main()

