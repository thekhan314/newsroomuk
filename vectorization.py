df_hate_words = pd.read_csv('data/raw_other/hateful_ngrams.csv')
df_hate_words.set_index('ngram',drop=True,inplace=True)
dict_hateweights = df_hate_words['prophate'].to_dict()


def main():
    """
    Converts all ftr files inside Data/ and write them as csv format in out/
    Folders are expected to exist.
    """
    file_list = glob("Data/*.ftr")
    from tqdm import tqdm

    for filename in file_list:
        outfilename = "out/" + filename[len('Data/'):]
        with open(f"{outfilename}.csv", 'wb') as f:
            print(f'* {filename}')
            df = feather.read_dataframe(filename)
            for _, r in tqdm(df.iterrows(), total=len(df)):
                converted = convert_article(r)
                # Write each article immediately in order to reduce memory pressure 
                csv = converted.to_csv(None, columns=['conv_id','timestamp', 'source', 'text'], header=False, index=False)
                f.write(bytes(csv, 'utf8'))
                f.write(b"\n")

if __name__ == "__main__":
    main()

#instantiate and fit the vectorizer
unlab_cvect = CountVectorizer(
    ngram_range=(1,4),
    tokenizer = tokenize,
    vocabulary = list(df_hate_words['ngram'])
)


unlab_vectors = unlab_cvect.fit(df_unlab['text'])
unlab_matrix = unlab_vectors.transform(df_unlab['text'])
df_vectors = pd.DataFrame(unlab_matrix.todense(),columns=unlab_vectors.get_feature_names())

#Gather hate counts and calculate hate scores
df_vectors = df_vectors.apply(lambda row:tally_counts_doc(row),axis=1,result_type='expand')
df_hatecounts = pd.concat([df_unlab,df_vectors],axis=1)
df_hatecounts['hate_hits'] = df_hatecounts['hate_hits'].astype('string')
df_hatecounts.to_csv(PROJECT_PATH + 'data/unlab_hatecounts.csv')

#Whittle down the vector dataframe to reduce size of DF by curtailing redundant data in the tables and allow for joins
df_hatecounts_lite = df_hatecounts.drop([
                                         'author',
                                         'text',
                                         'title',
                                         'subtitle',
                                         'timestamp',
                                         'source',
                                         'link'
                                         ],axis=1)
df_hatecounts_lite.to_sql('hatecounts',conn,if_exists='replace')
conn.commit()