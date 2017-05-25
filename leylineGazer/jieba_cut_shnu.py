# coding=utf-8
import jieba
import simplejson as json
import re
import pymongo
from pymongo import MongoClient

fileSegWordDonePath = './shnu/shnu_cut.txt'
# read the file by line
fileTrainRead = []
# fileTestRead = []

client = MongoClient('mongodb://localhost:27017/')
db = client.shnu_site
news_collection = db.News


with open('./shnu/shnu.json',encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
        news_collection.insert(line)

with open('./shnu/shnu0.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)


with open('./shnu/shnu1.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)


with open('./shnu/shnu2.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
with open('./shnu/shnu3.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
with open('./shnu/shnu4.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
with open('./shnu/shnu5.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
with open('./shnu/shnu6.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
with open('./shnu/shnu7.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
with open('./shnu/shnu8.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
with open('./shnu/shnu9.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
with open('./shnu/shnu00.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
with open('./shnu/shnu01.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
with open('./shnu/shnu02.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
with open('./shnu/shnu03.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
with open('./shnu/shnu04.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
with open('./shnu/shnu05.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)

with open('./shnu/shnu06.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
        news_collection.insert(line)
with open('./shnu/shnu07.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
        news_collection.insert(line)
with open('./shnu/shnu08.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
        news_collection.insert(line)
with open('./shnu/shnu09.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
        news_collection.insert(line)
with open('./shnu/shnu10.json', encoding='utf-8',errors='ignore') as f:
    for line in json.load(f):
        fileTrainRead.append(line)
        news_collection.insert(line)

# enableParallel
# 8vCPU
# jieba.enable_parallel(4)

# define this function to print a list with Chinese
def PrintListChinese(list):
    for i in range(len(list)):
        print(list[i]),

def inChinese(c):
    zhPattern = re.compile(u"[\u4e00-\u9fa5]+")
    try:
        return zhPattern.search(c) is not None
    except Exception as e:
        return None


# segment word with jieba
fileTrainSeg = []
allWordsCHN={}
allWordsENG={}

words_res_chn = []
words_res_eng = []
sum_wordsCHN = 0
sum_wordsENG = 0

count = 0
jieba.enable_parallel(4)
for i in fileTrainRead:
    l = jieba.cut(i['text'], cut_all=False)
    lst = list(l)
    for word in lst:
        if word is not None and len(word.replace(" ",""))>1:
            if inChinese(word):
                if allWordsCHN.get(word) is None:
                    allWordsCHN[word] = 1
                else:
                    allWordsCHN[word] += 1
            else:
                if allWordsENG.get(word) is None:
                    allWordsENG[word] = 1
                else:
                    allWordsENG[word] += 1
    fileTrainSeg.append([' '.join(lst)])
    count+=1
    print(str(count) + ' / ' + str(len(fileTrainRead)))



for w in allWordsCHN:
    sum_wordsCHN += allWordsCHN[w]

avg_wordsCHN = sum_wordsCHN / len(allWordsCHN)
for w in allWordsCHN:
    if allWordsCHN[w]>avg_wordsCHN:
        words_res_chn.append({w:allWordsCHN[w]})

for w in allWordsENG:
    sum_wordsENG += allWordsENG[w]

avg_WordsENG = sum_wordsENG / len(allWordsENG)
for w in allWordsENG:
    if allWordsENG[w]>avg_WordsENG:
        words_res_eng.append({w:allWordsENG[w]})


with open(fileSegWordDonePath,'wb') as fW:
    for i in fileTrainSeg:
        for ii in i:
            fW.write((ii + '\n').encode('utf-8'))

with open('topWordsCHN.json', 'w') as file:
    file.write(json.dumps(words_res_chn))

with open('topWordsENG.json', 'w') as file:
    file.write(json.dumps(words_res_eng))

def k(x):
    for (k,v) in x.items():
        return v

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
