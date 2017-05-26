import os

import sys
import pickle
import re
import logging
from gensim import similarities
import pymongo
from gensim import corpora
import jieba
from pymongo import MongoClient
from bson.json_util import dumps
from tld import get_tld

import os
from os.path import join


dest = "/Users/bytenoob/PycharmProjects/leylineGazer/leylineGazer/shnu/classed/"
files = os.listdir(dest)