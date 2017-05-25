import os

import sys

import logging
from gensim import similarities
import pymongo
from gensim import corpora
import jieba
from pymongo import MongoClient
from bson.json_util import dumps

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
client = MongoClient('mongodb://localhost:27017/')
db = client.shnu_site
news_collection = db.News
world = news_collection.find(no_cursor_timeout=True).sort("_id",1)
corpora_documents = []
raw_documents = []
with open('dump.txt','wb') as fW:
    for a in world:
        fW.write(dumps(a).encode())

