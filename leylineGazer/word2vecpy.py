#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

model = Word2Vec(LineSentence('cut2.txt'), size=400, window=5, min_count=5,workers=4)
model.save('cut2_1.bin')
model.wv.save_word2vec_format('cut2_2.txt', binary=False)

'multiprocessing.cpu_count()'
