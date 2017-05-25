import re
import word2vec
import simplejson as json
from gensim.models import Word2Vec
def inChinese(c):
    zhPattern = re.compile(u"[\u4e00-\u9fa5]+")
    try:
        return zhPattern.search(c) is not None
    except Exception as e:
        return None

pos = []
neg = []
sumchn=0
sumeng=0
with open('./POS.txt',encoding='utf-8',errors='ignore') as f:
    for line in f:
        pos.append(line.replace("\n",""))

with open('./NEG.txt',encoding='utf-8',errors='ignore') as f:
    for line in f:
        neg.append(line.replace("\n",""))

#model = Word2Vec.load('./shnu/w2v_0.bin')
#model = Word2Vec.load('./cut2_1.bin')

model = word2vec.load('./juice.bin')

count = 0
posres = []
negres = []
for w in pos:
    count+=1
#    if(count >100):
#        break
    try:
        indexes = model.most_similar(w)
    except Exception as e:
        continue
    countres=0
    for (k,v) in indexes:
        if k not in pos and inChinese(k) and len(k)>1 and k not in posres:
            countres+=1
            if(countres>2):
                break
            posres.append(k)
            print(k+' '+w)

for w in neg:
    count+=1
#    if(count >100):
#        break
    try:
        #indexes = model.most_similar(w)
        indexes = model.cosine(w)

    except Exception as e:
        continue
    countres=0
    '''
    for (k,v) in indexes:
        if k not in neg and inChinese(k) and len(k)>1 and k not in negres:
            countres+=1
            if(countres>2):
                break
            negres.append(k)
            print(k+' '+w)
    '''

    for i in indexes[0]:
        k=model.vocab[i]
        if k not in neg and inChinese(k) and len(k)>1 and k not in negres:
            countres+=1
            if(countres>2):
                break
            negres.append(k)
            print(k+' '+w)

with open('./POS_extended_juice.txt',mode="w",encoding='utf-8',errors='ignore') as f:
    for k in posres:
        f.write(k+'\n')

with open('./NEG_extended_juice.txt',mode="w",encoding='utf-8',errors='ignore') as f:
    for k in negres:
        f.write(k+'\n')