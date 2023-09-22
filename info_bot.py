#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:53:43 2023

@author: temuuleu
"""

import pandas as pd
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client.mydatabase


data_collection = db["data"]

data_list = list(data_collection.find())
df = pd.DataFrame(data_list)

instructions_collection = db["instruction_collection"]
Instructions = instructions_collection.find_one({"Instruction":"Search"})


all_search_instructions = list(instructions_collection.find({"Instruction": "Search"}))
# for doc in all_search_instructions:
#     print(doc)
    
columns = ['keyword',
           'n_clusters', 
           'negative',
           'neutral',
           'positive', 
           'summarized',
           'date', 
           'cluster_labels']
    
resutl_df = df.loc[df["number"] == 0,columns]
resutl_df  = resutl_df.rename(columns={"n_clusters":"strength"})
resutl_df  = resutl_df.sort_values(by=["keyword","strength"],ascending = False)







# all_data_collection = db["all_data"]


# # data_collection.drop()
# # all_data_collection.drop()


# data_list = list(all_data_collection.find())
# saved_df = pd.DataFrame(data_list)
