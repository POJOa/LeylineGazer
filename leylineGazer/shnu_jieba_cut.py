# coding=utf-8
import jieba
import simplejson as json
import re
import pymongo
from pymongo import MongoClient

output_path = './shnu/shnu_cut.txt'
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

for i in world:
    l = jieba.cut(i['text'], cut_all=False)
    lst = list(l)
    for word in lst:
        if word is not None and len(word.replace(" ",""))>1:
            if in_chinese(word):
                if chinese_words.get(word) is None:
                    chinese_words[word] = 1
                else:
                    chinese_words[word] += 1
            else:
                if non_chinese_words.get(word) is None:
                    non_chinese_words[word] = 1
                else:
                    non_chinese_words[word] += 1
    topics.append([' '.join(lst)])
    count+=1
    print(str(count) , ' / ' , str(world.count()))

with open(output_path,'wb') as f:
    for i in topics:
        for ii in i:
            f.write((ii + '\n').encode('utf-8'))



for w in chinese_words:
    freq_all_chinese_words += chinese_words[w]

freq_all_chinese_words = freq_all_chinese_words / len(chinese_words)
for w in chinese_words:
    if chinese_words[w]>freq_all_chinese_words:
        words_res_chn.append({w:chinese_words[w]})

for w in non_chinese_words:
    freq_all_non_chinese_words += non_chinese_words[w]

freq_avg_non_chinese_words = freq_all_non_chinese_words / len(non_chinese_words)
for w in non_chinese_words:
    if non_chinese_words[w]>freq_avg_non_chinese_words:
        words_res_eng.append({w:non_chinese_words[w]})

words_res_chn = sorted(words_res_chn, key=k)
words_res_chn.reverse()

words_res_eng = sorted(words_res_eng, key=k)
words_res_eng.reverse()

for w in words_res_chn:
    for (k,v) in w.items():
        if len(k)>1:
            print(w)

for w in words_res_eng:
    for (k,v) in w.items():
        if len(k)>1:
            print(w)


with open('./shnu/shnu_top_chinese_words.json', 'w') as file:
    file.write(json.dumps(words_res_chn))

with open('./shnu/shnu_top_non_chinese_words.json', 'w') as file:
    file.write(json.dumps(words_res_eng))



