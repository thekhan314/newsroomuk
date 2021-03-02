from bs4 import BeautifulSoup as bs
import pandas as pd
import pyarrow.feather as feather
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize
import string
import re
from nltk.stem.porter import *
import sqlite3



# Get data from DB
df_unlab = pd.read_sql('SELECT * FROM cleaned',conn,index_col='index')

df_hatevecs = pd.read_sql('SELECT * FROM hatecounts',conn,index_col='index')
df_hatevecs = df_hatevecs.drop(['timestamp','source','link'],axis=1) ## DELETE THIS AFTER RE-RUNNING THE PROCESSING NOTEBOOK 

df_hatecounts = df_unlab.merge(df_hatevecs,on='index')
df_hate_articles = df_hatecounts[df_hatecounts['hate_score'] > 0]
df_hate_articles = df_articles[['source','link','text','hate_score']]

df_utters = pd.DataFrame()

for index, row in df_utter_test.iterrows():
    tokens = sent_tokenize(row['text'])

    chunk = pd.DataFrame.from_dict({'sentence':tokens})
    chunk['source'] = row['source']
    chunk['link'] = row['link']
    chunk['article_index'] = index

    df_utters = pd.concat([df_utters,chunk])

sent_cvect = CountVectorizer(
    ngram_range=(1,4),
    tokenizer = tokenize,
    vocabulary = list(df_hate_words.index)
)


sent_vectors = sent_cvect.fit(df_utters['sentence'])
sent_matrix = sent_vectors.transform(df_utters['sentence'])
df_svectors = pd.DataFrame(sent_matrix.todense(),columns=sent_vectors.get_feature_names())

#Gather hate counts and calculate hate scores
df_svectors = df_svectors.apply(lambda row:tally_counts_doc(row),axis=1,result_type='expand')
df_shatecounts = pd.concat([df_utters,df_svectors],axis=1)