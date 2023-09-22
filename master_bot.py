#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 18:17:07 2023

@author: temuuleuimport pymongo
"""


#!pip install pymongo

import pymongo
from time import sleep
from threading import Thread


# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client.mydatabase

instructions_collection = db["instruction_collection"]


filter_ = {"Instruction": "Search"}

# Insert a document
# document = {"Instruction": "Search", "keywords":  ["Nvidia", "Paypal", "Immersion Corp","Virtual Reallity"], "time": "hourly"}
# instructions_collection.update_many(document)


"""TODO add new keywords from strong positive or negative assosiated words"""

# Define the changes you want to make
new_values = {
    "$set": {
        "keywords": ["Nvidia", "Paypal", "Immersion Corp", "Virtual Reality","Robotic", "Biotechnology", "Roblox", "Hydrogen", "Fusion"],
        "time": "hourly"
    }
}


# Update the document
instructions_collection.update_one(filter_, new_values)






# #how to search ?
# Instructions  = instructions_collection.find_one({"Instruction":"Search"})


# keywords_list = Instructions["keywords"]




        
# # Worker bot function
# def worker_bot(keyword):
#     print(f"Worker bot for {keyword} activated.")
#     while True:
#         # Your keyword-specific scraping logic here
#         print(f"Searching for {keyword}...")
#         sleep(600)  # Sleep for 10 minutes or adjust as needed



# # List to keep track of active bots
# active_bots = {}

# while True:
#     # Polling loop to check for instructions in MongoDB
#     instructions = instructions_collection.find_one({"type": "instruction"})
    
    
    
    
#     if instructions:
#         keywords = instructions.get("keywords", [])
        
#         # Activate new bots
#         for keyword in keywords:
#             if keyword not in active_bots:
#                 t = Thread(target=worker_bot, args=(keyword,))
#                 t.start()
#                 active_bots[keyword] = t
        
#         # Deactivate bots that are no longer needed
#         for keyword in list(active_bots.keys()):
#             if keyword not in keywords:
#                 # Terminate the thread (you might want a cleaner termination process)
#                 active_bots[keyword] = None
#                 del active_bots[keyword]
#                 print(f"Worker bot for {keyword} deactivated.")
    
#     # Sleep for 1 hour before checking again
#     sleep(3600)