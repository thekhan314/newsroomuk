from datetime import datetime
print(datetime.now())

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

        hate_score = 0
        hate_hits = {}

        for gram in master:
            if gram in dict_hateweights.keys():
                if gram in hate_hits:
                    hate_hits[gram] += 1 
                else:
                    hate_hits[gram] = 1
                hate_score += dict_hateweights[gram]
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
    ('hate_score',pyarrow.int64()),
    ('hate_hits',pyarrow.string())
    ])

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", default = "data/samp_utterances.parquet")
    parser.add_argument("--output", dest="output", default = "data/samp_out")
    app_args, pipeline_args = parser. parse_known_args()


    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        
        data = p | beam.io.parquetio.ReadFromParquet(app_args.input)
        output = data | beam.ParDo(WordExtractingDoFn())
        output | beam.io.parquetio.WriteToParquet(app_args.output,schema=schema)



run()



from datetime import datetime
print(datetime.now())