#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 19:46:57 2023

@author: temuuleu
"""

import pymongo
from time import sleep
from threading import Thread
import subprocess
import os
from datetime import datetime, timedelta
import time

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client.mydatabase

index_search = 0

# List of hours when the search should be triggered
search_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18,19,20,21,22,23]  # Example: 9 AM to 6 PM

index_search= 0

while True:
    try:
        current_hour = datetime.now().hour
        
        if current_hour in search_hours:
            
            index_search +=1
            instructions_collection = db["instruction_collection"]
            Instructions = instructions_collection.find_one({"Instruction": "Search"})

            if Instructions and "keywords" in Instructions:
                keywords_list = Instructions["keywords"]
                print("keywords_list")
                print(keywords_list)

                for keyword in keywords_list:
                    print(f"start searching : {keyword}  {index_search}")
                    os.system(f"nohup python search_bot.py {keyword} {index_search} &")
                    time.sleep(300)  # Sleep for a bit between keyword searches

            # Remove the hour from the list so it doesn't trigger again today
            search_hours.remove(current_hour)

            # If all hours are done for today, wait until tomorrow
            if not search_hours:
                now = datetime.now()
                # Calculate time until 9 AM tomorrow
                next_run = now + timedelta(days=1)
                next_run = next_run.replace(hour=9, minute=0, second=0, microsecond=0)
                sleep_time = (next_run - now).seconds
                print(f"All tasks done for today. Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
                # Reset the search hours for the next day
                search_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

        else:
            print("Not a search hour. Sleeping for a while.")
            time.sleep(5)  # Sleep for an hour and then check again

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)




