import os

import sys
import pickle

import logging
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




def clean(limit,skip,count):
    print('running')
    pId = count
    client = MongoClient('mongodb://localhost:27017/')
    db = client.shnu_site
    news_collection = db.News
    news_collection_raw = db.News_backup
    world = news_collection.find(no_cursor_timeout=True).sort("_id", 1).limit(limit).skip(skip)
    world_raw = news_collection_raw.find(no_cursor_timeout=True).sort("_id", 1)
    #jieba.enable_parallel(4)


    idmap = []


    for i in world_raw:
        idmap.append(i['_id'])

    similarity = similarities.Similarity.load('./shnu/similarity.bin')
    similarity.num_best = 5
    dictionary = corpora.Dictionary.load('./shnu/similarity_dict.txt')

    count = 0
    to_del = []
    print(str(pId),' - cutting text and generating corpus')
    for i in world:
        print(str(pId),' - processing no.', count , ' / ',str(limit))
        count+=1
        self_id = i['_id']

        if(self_id in to_del or news_collection.find_one({"_id":self_id}) is None):
            print(str(pId),' - '+ str(self_id) , ' deleted')
            continue
        s = similarity[dictionary.doc2bow(jieba.cut(i['text']))]
        for (k, v) in s:
            target_id = idmap[k]
            if v > 0.9999 and target_id != self_id:
                try:
                    news_collection.remove({"_id": target_id})
                    print(str(pId), ' - ' + str(target_id) + ' will be deleted because of similarity to ', str(self_id))
                    to_del.append(target_id)
                except Exception as e:
                    print(str(e))

    try:
        with open('pk-'+str(limit)+'-'+str(skip)+'.pkl', 'wb') as f:  # open file with write-mode
            picklestring = pickle.dump(to_del,f)  # serialize object
    except Exception as e:
        print(str(e))
        ''''
    count = 0
    all = len(to_del)
    for i in to_del:
        count += 1
        print(str(pId),' - deleting ',str(count),' / ',str(all))
        news_collection.remove({"_id": i})

    '''
    print(str(pId), ' - deleted')
    print(len(to_del))

    return len(to_del)


