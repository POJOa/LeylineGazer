# coding=utf-8
import jieba
import simplejson as json
import re
import pymongo
from tld import get_tld
from pymongo import MongoClient

output_path = './shnu/classed_cut/'
client = MongoClient('mongodb://localhost:27017/')
db = client.shnu_site
news_collection = db.News


# define this function to print a list with Chinese
def PrintListChinese(list):
    for i in range(len(list)):
        print(list[i]),

def in_chinese(c):
    zhPattern = re.compile(u"[\u4e00-\u9fa5]+")
    try:
        return zhPattern.search(c) is not None
    except Exception as e:
        return None

def k(x):
    for (k, v) in x.items():
        return v

# segment word with jieba
topics = []
chinese_words={}
non_chinese_words={}

words_res_chn = []
words_res_eng = []
freq_all_chinese_words = 0
freq_all_non_chinese_words = 0

count = 0

world = news_collection.find()
jieba.enable_parallel(4)
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
    with open(output_path+k+'.txt', 'w') as f:  # open file with write-mode
        for l in v:
            f.write(' '.join(list(jieba.cut(l, cut_all=False)))+'\n')
