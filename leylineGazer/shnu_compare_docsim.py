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

client = MongoClient('mongodb://localhost:27017/')
db = client.shnu_site
news_collection = db.News

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format=' %(process)d: %(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

world=[]
with open('./shnu/topic_sim_model.txt',encoding='utf-8',errors='ignore') as f:
    for line in f:
        world.append(line.replace("\n",""))


similarity = similarities.Similarity.load('./shnu/similarity.bin')
similarity.num_best = 5
dictionary = corpora.Dictionary.load('./shnu/similarity_dict.txt')

for i in world:
    s = similarity[dictionary.doc2bow(jieba.cut(i))]
    print('与以下文章相似：')
    print(i)
    print('\n')
    for (k, v) in s:
        if(v < 0.9):
            print('相似度：',v)
            target = news_collection.find().skip(int(k)).limit(1)
            print(target[0]['text'])
            print('\n')
            break


