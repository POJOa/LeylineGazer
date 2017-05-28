import os

import sys
import pickle
import re
import logging
from gensim import similarities
import pymongo
from gensim import corpora
import jieba
from pymongo import MongoClient
from bson.json_util import dumps
from tld import get_tld

client = MongoClient('mongodb://localhost:27017/')
db = client.shnu_site
news_collection = db.News
news_collection_raw = db.News_backup
world = news_collection.find(no_cursor_timeout=True).sort("_id", 1)
world_raw = news_collection_raw.find(no_cursor_timeout=True).sort("_id", 1)
#jieba.enable_parallel(4)


idmap = []
groups = {}
for i in world:
    try:
        dom_obj = get_tld(i['link'],as_object=True)
        dom = dom_obj.subdomain
    except:
        dom = 'unknown'
    if groups.get(dom) is None:
        groups[dom]=[]
    groups[dom].append(i['text'])

for (k,v) in groups.items():
    if k is None or len(k)<1:
        k='empty'
    with open('./shnu/classed/'+k+'.txt', 'w') as f:  # open file with write-mode
        for p in v:
            f.write(p+'\n')
