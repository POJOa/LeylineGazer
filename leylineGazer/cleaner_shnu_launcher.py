import os
from cleaner_shnu_1 import clean

import sys
import pickle

import logging
from multiprocessing.pool import Pool
import multiprocessing
from gensim import similarities
import pymongo
from gensim import corpora
import jieba
from pymongo import MongoClient
from bson.json_util import dumps



def err(e):
    print(e)




def main():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.shnu_site
    news_collection = db.News
    world_num_max = 4
    world_count = news_collection.find(no_cursor_timeout=True).sort("_id", 1).count()
    #world_count=1000
    each_world_count = int(world_count / world_num_max)
    worlds =[]
    print(world_count)
    print(each_world_count)

    for world_num in range(0,world_num_max-2):
        worlds.append((each_world_count,each_world_count*world_num))
    worlds.append((world_count-each_world_count*(world_num_max-1),each_world_count*(world_num_max-1)))

    p=Pool(world_num_max)
    for (limit,skip) in worlds:
        p.apply_async(clean,args=(limit,skip,),error_callback=err)

    p.close()
    p.join()

if __name__ == "__main__":
    main()

