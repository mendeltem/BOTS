#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:24:16 2023

@author: temuuleu
"""

# bot.py

import sys
import time  # Import the time module

import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers.util import cos_sim
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

from duckduckgo_search import DDGS
import pandas as pd

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from transformers import pipeline


import pymongo
#pip install duckduckgo_search
#pip install pymongo
#!pip install kneed
from kneed import KneeLocator
from sklearn.cluster import KMeans

import importlib
import os 
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.cluster import KMeans

from datetime import datetime
import transformers
transformers.logging.set_verbosity(transformers.logging.ERROR)

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax


def extract_hyperlinks(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    """Extract hyperlinks from a BeautifulSoup object

    Args:
        soup (BeautifulSoup): The BeautifulSoup object
        base_url (str): The base URL

    Returns:
        List[Tuple[str, str]]: The extracted hyperlinks
    """
    return [
        (link.text, urljoin(base_url, link["href"]))
        for link in soup.find_all("a", href=True)
    ]


def check_local_file_access(url: str) -> bool:
    """Check if the URL is a local file

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is a local file, False otherwise
    """
    local_prefixes = [
        "file:///",
        "file://localhost/",
        "file://localhost",
        "http://localhost",
        "http://localhost/",
        "https://localhost",
        "https://localhost/",
        "http://2130706433",
        "http://2130706433/",
        "https://2130706433",
        "https://2130706433/",
        "http://127.0.0.1/",
        "http://127.0.0.1",
        "https://127.0.0.1/",
        "https://127.0.0.1",
        "https://0.0.0.0/",
        "https://0.0.0.0",
        "http://0.0.0.0/",
        "http://0.0.0.0",
        "http://0000",
        "http://0000/",
        "https://0000",
        "https://0000/",
    ]
    return any(url.startswith(prefix) for prefix in local_prefixes)

def is_valid_url(url: str) -> bool:
    """Check if the URL is valid

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    
# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def fetch_news_to_dataframe(keywords):
    
    keywords="NVida"
    import os
    print(os.environ.get('HTTP_PROXY'))
    print(os.environ.get('HTTPS_PROXY'))

    
    with DDGS() as ddgs:
        ddgs_news_gen = ddgs.news(
            keywords,
            region="wt-wt",
            safesearch="off",
            timelimit="m",
        )
        
        # Create a list to store the results
        results = []
        
        for r in ddgs_news_gen:
            results.append({
                'date': r['date'],
                'title': r['title'],
                'body': r['body'],
                'image': r['image'],
                'source': r['source']
            })
        
        # Convert the list of results to a DataFrame
        df = pd.DataFrame(results, columns=['date', 'title', 'body', 'image', 'source'])
        
        return df
    




def find_optimal_clusters(embeddings, max_clusters=10):
    inertia_values = []
    cluster_range = range(1, max_clusters+1)

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters).fit(embeddings)
        inertia_values.append(kmeans.inertia_)

    # Use KneeLocator to find the elbow point
    kn = KneeLocator(cluster_range, inertia_values, curve='convex', direction='decreasing')
    optimal_clusters = kn.elbow

    return optimal_clusters


def cluster_words(embeddings, n_clusters):
    kmeans_model = KMeans(n_clusters=n_clusters, n_init=10)  # Add n_init parameter
    cluster_labels = kmeans_model.fit_predict(embeddings)
    cluster_centers = kmeans_model.cluster_centers_

    return cluster_labels, cluster_centers


def chunked_summarization(text, chunk_size=1000):
    # Split the text into chunks of the specified size
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Summarize each chunk and store the results in a list
    summarized_chunks = [summarizer(chunk, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
    
    # Concatenate the summarized chunks
    summarized_text = ' '.join(summarized_chunks)
    
    return summarized_text


def fetch_and_combine_data(keyword):
    new_df = fetch_news_to_dataframe(keyword)
    new_df["keyword"] = keyword
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client.mydatabase
    all_data_collection = db["all_data"]
    data_list = list(all_data_collection.find())
    saved_df = pd.DataFrame(data_list)
    combined_df = pd.concat([new_df, saved_df], ignore_index=True)
    new_info_df = new_df[~new_df['title'].isin(saved_df['title'])]
    combined_df.drop_duplicates(subset='title', keep='first', inplace=True)
    if '_id' in combined_df.columns:
        combined_df.drop('_id', axis=1, inplace=True)
    all_data_collection.delete_many({})
    all_data_collection.insert_many(combined_df.to_dict('records'))
    return new_info_df


def sentiment_scores(text, MODEL="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    # Constants

    # Load tokenizer, config, and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    # Preprocess and encode the text
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    
    # Get scores and apply softmax
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Sort scores and labels
    ranking = np.argsort(scores)[::-1]
    sorted_labels = [config.id2label[ranking[i]] for i in range(scores.shape[0])]
    sorted_scores = [scores[ranking[i]] for i in range(scores.shape[0])]
    
    # Create a dictionary with sentiment labels as keys
    sentiment_dict = {label: score for label, score in zip(sorted_labels, sorted_scores)}
    
    return sentiment_dict





def preprocess(text):
    new_text = [ '@user' if t.startswith('@') and len(t) > 1 else t for t in text.split() ]
    new_text = [ 'http' if t.startswith('http') else t for t in new_text ]
    return " ".join(new_text)


def sentiment_scores(text, MODEL="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    # Preprocess the text
    text = preprocess(text)
    
    # Load tokenizer, config, and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    # Encode the text
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    
    # Get scores and apply softmax
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    print(f"scores {scores}")
    
    # Sort scores and labels
    ranking = np.argsort(scores)[::-1]
    sorted_labels = [config.id2label[ranking[i]] for i in range(scores.shape[0])]
    sorted_scores = [scores[ranking[i]] for i in range(scores.shape[0])]
    
    # Return a dictionary with sentiment labels as keys
    return {label: score for label, score in zip(sorted_labels, sorted_scores)}

        
# text = "Covid cases are increasing fast!"
# result = sentiment_scores(text)
# print(result)



def apply_sentiment_and_summary(df):
    df['negative'] = df['body'].apply(lambda x: sentiment_scores(x)['negative'])
    df['neutral'] = df['body'].apply(lambda x: sentiment_scores(x)['neutral'])
    df['positive'] = df['body'].apply(lambda x: sentiment_scores(x)['positive'])
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    df['summary'] = df['body'].apply(lambda x: summarizer(x, max_length=len(x.split()), min_length=0, do_sample=False)[0]["summary_text"])
    return df


def cluster_and_save(df, number):
    print(f"loading model..")
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    print(f"loading model finished")
    sentences_list = df["summary"]
    tokens = model.encode(sentences_list)
    print(f"created tokens: {len(tokens)}")
    n_clusters = find_optimal_clusters(tokens)
    print(f"performing kmeans with k={n_clusters}")
    cluster_labels, cluster_centers = cluster_words(tokens, n_clusters)
    df["cluster_labels"] = cluster_labels
    df['n_clusters'] = df.groupby('cluster_labels')['cluster_labels'].transform('size')
    clustered_df = df.groupby('cluster_labels').agg({
        'summary': ' '.join,
        'n_clusters': 'first',
        'negative': 'mean',
        'neutral': 'mean',
        'positive': 'mean'
    }).reset_index()
    clustered_df['negative'] = clustered_df['negative'].round(2)
    clustered_df['neutral'] = clustered_df['neutral'].round(2)
    clustered_df['positive'] = clustered_df['positive'].round(2)
    clustered_df['summarized'] = clustered_df['summary'].apply(lambda x: chunked_summarization(x, chunk_size=1000))
    current_date = datetime.now().strftime('%Y-%m-%d')
    clustered_df['date'] = current_date
    clustered_df['number'] = number
    clustered_df['keyword'] = df["keyword"]
    data_dict = clustered_df.to_dict(orient='records')
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client.mydatabase
    data_collection = db["data"]
    data_collection.insert_many(data_dict)
    print("data summarization saved")
    
    

def main(keyword, number=0):
    df = fetch_and_combine_data(keyword)
    
    #df = combined_df
    if not df.empty:
        sentinemental_df = apply_sentiment_and_summary(df)
        sentinemental_df["keyword"] = keyword
        cluster_and_save(sentinemental_df, number)
    
    
    

# def main(keyword, number):
    
#     keyword = "mendel"

#     new_df = fetch_news_to_dataframe(keyword)
    
#     # 2. Connect to MongoDB and retrieve the saved data.
#     client = pymongo.MongoClient("mongodb://localhost:27017/")
#     db = client.mydatabase
#     all_data_collection = db["all_data"]
#     data_list = list(all_data_collection.find())
#     saved_df = pd.DataFrame(data_list)
    
#     # 3. Concatenate the new data with the saved data.
#     combined_df = pd.concat([new_df, saved_df], ignore_index=True)
    
#     # 4. Filter out any duplicate news based on the title.
#     # Assuming 'title' is the column name that contains the news title.
#     combined_df.drop_duplicates(subset='title', keep='first', inplace=True)
    
#     # Drop the _id column to let MongoDB generate new unique IDs.
#     if '_id' in combined_df.columns:
#         combined_df.drop('_id', axis=1, inplace=True)
    
#     # 5. Save the updated DataFrame back to MongoDB.
#     # First, clear the old data in the collection.
#     all_data_collection.delete_many({})
#     # Then, insert the new data.
#     all_data_collection.insert_many(combined_df.to_dict('records'))
        
    


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bot.py <keyword> <number>")
        sys.exit(1)

    keyword_arg = sys.argv[1]
    number_arg = int(sys.argv[2])  # Convert the second argument to an integer

    keyword_arg = "Nvidia"
    main(keyword_arg, number_arg)



