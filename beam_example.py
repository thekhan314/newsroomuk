from typing import List, Optional
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import *
import string
import re
import apache_beam as beam
import argparse
from apache_beam.options.pipeline_options import PipelineOptions
from nltk.stem.porter import *
import warnings
warnings.filterwarnings('ignore') 

from datetime import datetime
print(datetime.now())

df_hate_words = pd.read_csv('data/raw_other/hateful_ngrams.csv')
df_hate_words.set_index('ngram',drop=True,inplace=True)
dict_hateweights = df_hate_words['prophate'].to_dict()
hate_list = list(df_hate_words.index)



stemmer = PorterStemmer()

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    #tokens = re.split("[^a-zA-Z]*", tweet.lower())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

unlab_cvect = CountVectorizer(
    ngram_range=(1,4),
    vocabulary = hate_list,
    tokenizer = tokenize
)
def tally_counts_doc(row):
    row2 = row[row > 0]
    score = 0
    hits = {}
    for index,val in row2.items():
        hits[index]=val
        hit = dict_hateweights[index] * val
        score += hit 
    row['hate_score'] = score
    row['hate_hits'] = hits

    return row

class WordExtractingDoFn(beam.DoFn):
    def process(self,element):
        unlab_vectors = unlab_cvect.fit([element['text']])
        unlab_matrix = unlab_vectors.transform([element['text']])
        df_vectors = pd.DataFrame(unlab_matrix.todense())
        df_vectors = df_vectors.apply(lambda row:tally_counts_doc(row),axis=1,result_type='expand')

        yield df_vectors[['hate_score','hate_hits']].to_dict()

  


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", default = "data/samp_utterances.parquet")
    parser.add_argument("--output", dest="output", default = "data/")
    app_args, pipeline_args = parser. parse_known_args()


    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        
        data = p | beam.io.parquetio.ReadFromParquet(app_args.input)
        output = data | beam.ParDo(WordExtractingDoFn())
        output | beam.io.textio.WriteToText('data/samp_output.csv')



run()

from datetime import datetime
print(datetime.now())