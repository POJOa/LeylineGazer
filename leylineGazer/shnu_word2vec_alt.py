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





program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

#model = Word2Vec(LineSentence('./shnu/shnu_cut.txt'), size=400, window=15, min_count=5, negative=10, workers=4)
model = Word2Vec(LineSentence('./shnu/classed_cut/all.txt'), size=400, window=5, min_count=1, workers=4)
model.save('./shnu/shnu_w2v_alt_trained_400_uncut_lined.bin')
#model.wv.save_word2vec_format('cut2_2.txt', binary=False)