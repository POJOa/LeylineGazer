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
with open('./EXP.txt',encoding='utf-8',errors='ignore') as f:
    for line in f:
        pos.append(line.replace("\n",""))

model_shnu = Word2Vec.load('./shnu/shnu_w2v_alt.bin')
model_blog = Word2Vec.load('./cut2_1.bin')
model_sohu = word2vec.load('./juice.bin')

count = 0
posres = []
negres = []
for w in pos:
    count+=1
#    if(count >100):
#        break
    try:
        indexes_shnu = model_shnu.most_similar(w)
    except Exception as e:
        indexes_shnu = []
        #print('shnu model miss for word '+w)

    try:
        indexes_blog = model_blog.most_similar(w)
    except Exception as e:
        indexes_blog = []
        #print('blog model miss for word ' + w)

    try:
        indexes_sohu = model_sohu.cosine(w)
    except Exception as e:
        indexes_sohu = []
        #print('sohu model miss for word ' + w)

    print('上海师大校园网 - ' + w + '：')
    str_shnu=''
    for (k,v) in indexes_shnu:
        str_shnu+=(k+' ')
    print(str_shnu)

    print('\n搜狐新闻 - ' + w+ '：')
    str_sohu=''
    for i in indexes_sohu[0]:
        str_sohu+=(model_sohu.vocab[i]+' ')
    print(str_sohu)

    print('\n程序员博客 - ' + w + '：')
    str_blog=''
    for (k,v) in indexes_blog:
        str_blog+=(k+' ')
    print(str_blog)
    print('---------------------------------')
