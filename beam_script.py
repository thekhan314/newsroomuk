from typing import List, Optional
import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import *
import string
import re
import apache_beam as beam

import argparse
from apache_beam.options.pipeline_options import PipelineOptions
from nltk.stem.porter import *
import warnings
warnings.filterwarnings('ignore') 
import pyarrow

in_out = {'input':'gs://nroom_utters/input/combined_utterances.parquet','output':'gs://nroom_utters/output/','runner':'DataflowRunner'}
#in_out = {'input':'data/samp_utterances.parquet','output':'data/out','runner':'DirectRunner'}

p_options = {
    'project':'gcpnroom',
    'region':'us-west4',
    'runner':in_out['runner'],
    'input':'gs://nroom_utters/input/combined_utterances.parquet',
    'output': 'gs://nroom_utters/output/',
    'staging_location':'gs://nroom_utters/staging',
    'temp_location':'gs://nroom_utters/tmp',
    'save_main_session':True,
    'setup_file': './setup.py',
    'service_account_email':'nroomservice@gcpnroom.iam.gserviceaccount.com'
}

pipeline_options = PipelineOptions(flags=[], **p_options)

stemmer = PorterStemmer()

df_hate_words = pd.read_csv('data/raw_other/hateful_ngrams.csv')
df_hate_words.set_index('ngram',drop=True,inplace=True)
dict_hateweights = df_hate_words['prophate'].to_dict()
hate_list = list(df_hate_words.index)


def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    #tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    
    tokens = re.split("[^a-zA-Z]*", tweet.lower())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens
    
def ngrams(tokens,n):
    ngrams = zip(*[tokens[i:] for i in range(n)])
    list_gram = [" ".join(ngram) for ngram in ngrams]
    return list_gram



class WordExtractingDoFn(beam.DoFn):
    def process(self,element):
        master = []
        tokens = tokenize(element['text'])

        for n in range(2,4):
            current = ngrams(tokens,n)
            master += current
          

        master += tokens

        hate_score = 0.0
        hate_hits = {}

        for gram in master:
            if gram in dict_hateweights.keys():
                if gram in hate_hits:
                    hate_hits[gram] += 1 
                else:
                    hate_hits[gram] = 1
                hate_score = hate_score + float(dict_hateweights[gram])

        element['hate_score'] = hate_score
        element['hate_hits'] = str(hate_hits)

        if hate_score > 0:
            yield element
        else:
            return

schema = pyarrow.schema([
    ('conv_id', pyarrow.string()),
    ('timestamp',pyarrow.int64()),
    ('source',pyarrow.string()),
    ('text',pyarrow.string()),
    ('hate_score',pyarrow.float64()),
    ('hate_hits',pyarrow.string())
    ])

import apache_beam as beam

def run():

    with beam.Pipeline(options=pipeline_options) as p:
        
        data = p | beam.io.parquetio.ReadFromParquet(in_out['input'])
        output = data | beam.ParDo(WordExtractingDoFn())
        output | beam.io.parquetio.WriteToParquet(in_out['output'],schema=schema)


if __name__ == "__main__":
    run()


