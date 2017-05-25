import simplejson as json
from gensim.models import Word2Vec

words_res_chn = []
words_res_eng = []
sumchn=0
sumeng=0
with open('topWordsCHN_sorted.json', encoding='utf-8') as f:
    words_res_chn = json.load(f)

with open('topWordsENG_sorted.json', encoding='utf-8') as f:
    words_res_eng = json.load(f)

model = Word2Vec.load('cut2_1.bin')
count = 0
for w in words_res_chn:
    if(count<=10):
        for (k,v) in w.items():
            indexes = model.most_similar(k)
            print('与'+k+'相似的词')
            print(indexes)
            print('\n')
        count+=1
    else:
        break
