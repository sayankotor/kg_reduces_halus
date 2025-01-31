from threading import Thread, BoundedSemaphore
import time, random
from queue import Queue
import json
import re
from html2text import html2text as htt
import wikitextparser as wtp

import urllib.request

from wikimapper import WikiMapper

import os
cur_path = os.path.dirname(os.path.abspath(__file__))
DB_PATH = cur_path + "/data_for_dump/index_enwiki-latest.db"

mapper = WikiMapper(DB_PATH)

 


def get_entity_by_target(target):
    url = "https://en.wikipedia.org/wiki/" + target
    url = url.replace(" ", "_")
    return mapper.url_to_id(url)
    

def get_entity_by_word(word):
    prefix = 'https://en.wikipedia.org/w/api.php?action=query&format=json&prop=pageprops&titles='
    query_string = prefix+word
    #print ("\n\n\n")
    #print (query_string)
    query_string = query_string.replace(" ", "%20")
    urllib.request.urlretrieve(query_string, 'page.json')

    try:
        with open('page.json') as json_data:
            #print ("open")
            d = json.load(json_data)
            json_data.close()
            item = d['query']['pages']
            number = next(iter(item))
            #print (item[number]['pageprops']['wikibase_item'])
            return item[number]['pageprops']['wikibase_item']

    except Exception as oops:
            print("exeption in retireving entity", oops)
            pass
    return -1

def get_spans(link):
    return link.span

def get_title(link):
    return link.title
 

def dewiki(text):
    wt = wtp.parse(text)   
    text = wt.plain_text(replace_wikilinks=True)  # wiki to plaintext 
    links = wt.wikilinks
    #list_of_entity_names = [link.text for link in links]
    list_of_spans = [link.span for link in links]
    list_of_titles = [link.title for link in links]
    list_of_targets = [link.target for link in links]
    text = htt(text)  # remove any HTML
    text = text.replace('\\n',' ')  # replace newlines
    text = re.sub('\s+', ' ', text)  # replace excess whitespace
    return text, list_of_titles, list_of_targets, list_of_spans


def analyze_chunk(text):
    try:
        if '<redirect title="' in text:  # this is not the main article
            return None
        if '(disambiguation)' in text:  # this is not an article
            return None
        else:
            title = text.split('<title>')[1].split('</title>')[0]
            title = htt(title)
            if ':' in title:  # most articles with : in them are not articles we care about
                return None
        serial = text.split('<id>')[1].split('</id>')[0]
        content = text.split('</text')[0].split('<text')[1].split('>', maxsplit=1)[1]
        content, titles, targets, spans = dewiki(content)
        ent_ids = []
        existed_spans = []
        ent_names = []
        for ind, target in enumerate(targets):
            try:
                #entity = get_entity_by_word(title)
                entity = get_entity_by_target(target)
                ent_ids.append(entity)
                existed_spans.append(spans[ind])
                ent_names.append(target)
            except Exception as oops:
                print("exeption in creating entity", oops)
                pass
        entity_dict = {'ents':ent_ids, 'spans':existed_spans, 'ent_names':ent_names}   

        return {'title': title.strip(), 'text': content.strip(), 'id': serial.strip(), 'ents':entity_dict}
    except Exception as oops:
        print(oops)
        return None

import sys

threadLimiter = BoundedSemaphore(100)

class MyThread(Thread):

    def run(self):
        threadLimiter.acquire()
        try:
            self.Executemycode()
        finally:
            threadLimiter.release()

    def Executemycode(self, article, savedir):
        save_article(article, savedir)
        # <your code here>

def save_article(article, savedir):
    doc = analyze_chunk(article)
    if doc:
        print('SAVING:', doc['title'])
        filename = doc['id'] + '.json'
        with open(savedir + filename, 'w', encoding='utf-8') as outfile:
            json.dump(doc, outfile, sort_keys=True, indent=1, ensure_ascii=False)
                
                
def process_file_text(filename, savedir):
    article = ''
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            if '<page>' in line:
                article = ''
            elif '</page>' in line:  # end of article
                #save_article(article, savedir)

                #MyThread(target=save_article, args=(jobs, article, savedir))
                try:
                    th = MyThread()
                    th.Executemycode(article, savedir)
                except Exception as oops:
                    print("in saving: ", oops)
                    pass
            else:
                article += line
    print("all done")
    


def process_file_text1(filename, savedir):
    article = ''
    jobs = Queue()
    for i in range(100):
        jobs.put(i)
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            if '<page>' in line:
                article = ''
            elif '</page>' in line:  # end of article
                #save_article(article, savedir)
                worker = Thread(target=save_article, args=(jobs, article, savedir))
                worker.start()
            else:
                article += line
    print("waiting for queue to complete", jobs.qsize(), "tasks")
    jobs.join()
    print("all done")
