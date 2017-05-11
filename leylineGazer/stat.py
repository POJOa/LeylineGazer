#-*- coding: UTF-8 -*-

from snownlp import SnowNLP
from snownlp import seg
from tld import get_tld
import simplejson as json

sents = json.load(open('resSents.json'))
words = json.load(open('resWords.json'))
sum_words = 0
sents_res = []
words_res = []

for domain in sents:
    sents_arr = sents[domain]
    sents_res.append({domain:sum(sents_arr) / len(sents_arr)})

for w in words:
    sum_words += words[w]

avg_words = sum_words / len(words)
for w in words:
    if words[w]>avg_words:
        words_res.append({w:words[w]})



with open('statsSents.json', 'w') as file:
    file.write(json.dumps(sents_res))
with open('topWords.json', 'w') as file:
    file.write(json.dumps(words_res))
