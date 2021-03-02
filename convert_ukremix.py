#!/usr/bin/env python3
from typing import List, Optional
import feather
import pandas as pd
from glob import glob
from dateutil.parser import isoparse
from hashlib import sha1
from bs4 import BeautifulSoup
import bs4

def cleanup_tags(entry: str) -> str:
    """
    Returns the text contained within some HTML markup
    """
    if not entry:
        return entry

    soup = BeautifulSoup(entry, features='lxml')
    return soup.body.get_text()

def parse_array(entry: Optional[str]) -> List[str]:
    """
    Parses an array that has been stringified with an inner HTML value.
    Returns a python list where each element has tags stripped through beautifulsoup.


    # Example
    lst = parse_array("[<p>first</p>, <p>second</p>, <p>third</p>]")
    self.assertEquals(lst, ["first", "second", "third"])

    # Empty list
    lst = parse_array("[]")
    self.assertEquals(lst, [])

    # Single ement
    lst = parse_array("[<p>foo</p>]")
    self.assertEquals(lst, ["foo"])

    # Single element without markup
    lst = parse_array("[foo]")
    self.assertEquals(lst, ["foo"])
    
    """
    if not entry:
        return []

    ret = []
    soup = BeautifulSoup(entry, features='lxml')
    separators = {'[',',',']'}
    container = soup.html.body.contents
    
    if len(container) == 1 and len(container[0].contents) > 0:
        container = container[0].contents
        
    for elm in container:
        text = None
        if isinstance(elm, bs4.element.Tag):
            text = elm.get_text()
        elif isinstance(elm, bs4.element.NavigableString):
            text = str(elm)
        else:
            raise RuntimeError(f"Unexpected type: {type(elm)}")
        
        text = text.strip()
        if text not in separators:
            ret.append(text)
    if len(ret) == 1:
        if ret[0] == "[]":
            return []
        ret[0] = ret[0].strip("[]")

    return ret


def convert_article(row: pd.Series) -> pd.DataFrame:
    """
    Converts a single article into a DataFrame representing HumanFirst's conversation format

    This keeps a lot of the article's metadata as "expert input" in order to be able to view them contextually.

    """
    ## HumanFirst's CSV Columns:
    # 1: conversation id
    # 2: utterance ts, in miliseconds unix epoch
    # 3: input source ("client" or "expert") 
    # 4: input text

    link = row['link']
    timestamp = row['timestamp']
    article_title = parse_array(row['article_title'])
    article_subtitle = parse_array(row['article_subtitle'])
    article_text = parse_array(row['article_text'])
    author_name = parse_array(row['author_name'])

    # Compute a sha1 hash of the url, and use it as "conversation id"
    conv_id = sha1(bytes(link, 'utf8')).digest().hex()
    ts = 0
    if timestamp:
        try:
            ts = int(isoparse(timestamp).timestamp() * 1000)
        except:
            pass

    # Add metadata about the article as expert inputs
    meta_text = [
        f"Article URL: {link}",
    ]

    article = []

    for title in article_title:
        meta_text.append(f"Title: {title}")
    for subtitle in article_subtitle:
        meta_text.append(f"Subtitle: {subtitle}")
    for author in author_name:
        meta_text.append(f"Author: {author}")

    if article_text:
        article.extend(article_text)
    
    expert = pd.DataFrame([
        { 'conv_id': conv_id,  'timestamp': ts, 'source': 'expert', 'text': text }
        for text in meta_text if len(text.strip()) > 0
    ])

    client = pd.DataFrame([
        { 'conv_id': conv_id,  'timestamp': ts, 'source': 'client', 'text': text }
        for text in article if len(text.strip()) > 0
    ])

    return expert.append(client)
    

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
