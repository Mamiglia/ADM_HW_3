from urllib import request
from bs4 import BeautifulSoup
from IPython.display import clear_output
import os
import re
import numpy as np
import pandas as pd

# from func.scraper import crawl_urls, setup_folders, display_bar, download_pages, sanitize, select_one, select, parse_page, extract_data, generate_queries


def crawl_urls(websites, filename='urls.txt'):
    '''Collects the first 7200 urls from the Atlas Obscura website'''
    f = open(filename, 'w')
    for web in websites:
        req = request.Request(w)
        req.add_header('User-Agent', 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)')
        with request.urlopen(req) as response:
            soup = BeautifulSoup(response.read(), features="lxml")
            for a in soup.body.find_all('a', attrs={'class' : 'content-card'}):
                f.write('https://www.atlasobscura.com' + a.get('href')+'\n')
        # To dodge Error 429 : Too Many Requests
        time.sleep(1) 
    f.close()
    
def setup_folders(N=400, base='page', root='data'):
    '''Creates the basic folders'''
    for i in range(1,N+1):
        os.makedirs(f'{root}/{base}{i}', exist_ok=True) 
    return f'{root}/{base}'

def display_bar(partial, total):
    '''Displays a loading bar'''
    perc = 100*partial//total
    clear_output(wait=True)
    print(f'[{"="*(perc//2):50}] {perc}% : {partial}')

def download_pages(src, start=1, end=7201, base_dir = '', naming = lambda i: str(i), sleep=1,verbose=False):
    '''Downloads pages from src'''
    for i,url in enumerate(src[start-1:end], start=start):
        if verbose and i%30==0:
            display_bar(i, end)
        req = request.Request(url)
        req.add_header('User-Agent', 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0) Plz I need to study')
        place_name = f'{base_dir}/{naming(i)}.html'
        if os.path.exists(place_name):
            print(f'file already exists: {i} \t {place_name}{" ":50}')
            continue
        try:
            with request.urlopen(req) as response:
                place_i = open(place_name,'wb')
                place_i.write(response.read())
                place_i.close()
                time.sleep(sleep)
        except Exception as e:
            print(e)
            print(f"Crawling error at {i}th iteration")
            time.sleep(5)
    return end

def sanitize(s):
    '''Sanitizes a string by removing all unwanted charachters'''
    return s.strip().replace('\t', '').replace('"', '')

def select_one(html, query):
    '''Uses a CSS selector to select exactly one element, in case of failure returns "NA"''' 
    res = html.select_one(query)
    if res is None:
        return 'NA'
    return sanitize(res.text)

def select(html, query, func = lambda x: x, prefunc = None):
    '''Selects any number of elements using a CSS selector, then applies a function
        :func: function to apply pieceweise on the text elements
        :prefunc: function to apply on the elements list as a whole
        :returns: "NA" in case of error or empty list 
    '''
    res = html.select(query)
    if res is None or len(res)==0:
        return 'NA'
    try:
        if prefunc is not None:
            return [sanitize(x) for x in prefunc(res)]
        return [func(sanitize(x.text)) for x in res]
    except Exception as e:
        print(query, e)
        print(res)
        return 'NA'

LOC = r'^<div>(.*)\s*<div>'

def parse_page(page, url):
    '''Parses a page of Atlas Obscura and extracts data from it'''
    data = dict()
    html = BeautifulSoup(page, features="lxml")
    data['placeName'] = select_one(html, 'h1.DDPage__header-title')
    data['placeTags'] = set(select(html, 'a.itemTags__link', str.strip))
    if data['placeTags'] == {'A','N'}:
        data['placeTags'] = 'NA'
    data['numPeopleVisited'], data['numPeopleWant'], *_ = select(html, 'div.title-md.item-action-count', int)
    data['placeDesc'] = "\n".join(select(html, 'div.DDP__body-copy p'))
    data['placeShortDesc'] = select_one(html, 'h3.DDPage__header-dek')
    data['placeNearby'] = select(html, 'div.DDPageSiderailRecirc__item-title')
    data['placeAddress'] =  select(html, 'address.DDPageSiderail__address>div', 
            prefunc = lambda l: re.match(LOC, str(l[0])).groups())[0].replace('<br/>', '; ')
    data['placeAlt'], data['placeLng'] = [float(item) for item in select_one(html, 'div.DDPageSiderail__coordinates').split(',')]
    data['placeEditors'] = set(select(html, 'div.DDPContributorsList>a.DDPContributorsList__contributor'))
    list_editors = select(html, 'li.DDPContributorsList__item span')
    data['placeEditors'].update(list_editors if list_editors != 'NA' else {})
    data['placePubDate'] = select_one(html, 'div.DDPContributor__name')
    data['placeRelatedLists'] = select(html, 'a[data-gtm-content-type="List"] h3>span')
    data['placeRelatedPlaces'] = select(html, 'div[data-gtm-template="DDP Footer Recirc Related"] span')
    data['placeUrl'] = url
    data['placeZone'] = select_one(html, 'div.DDPage__header-place-location')
    return data

def extract_data(filename='urls.txt', start=1, end=7201, verbose=True):
    '''Parses each page and saves the data to a place_i.tsv'''
    os.makedirs('refined_data', exist_ok=True)
    url_file = open(filename, 'r')
    for i,url in enumerate(url_file.readlines()[start-1:end], start=start):
        if verbose and i%50==0:
            display_bar(i, end)
        place_name = f'data/page{(i-1)//18+1}/place_{(i-1)%18}.html'
        if not os.path.exists(place_name):
            raise Exception(f'file not found: {i} \t {place_name}{" ":50}')
        with open(place_name, 'r') as html:
            data_i = parse_page(html, url.strip())
            tab_values = '\t'.join(str(item).replace('\n', '\\n') for item in data_i.values())
            with open(f'refined_data/place_{i}.tsv', 'w') as out:
                out.write(tab_values)
    url_file.close()

    '''Aggregates data from different pages_i.tsv in a single file'''
    with open('places.tsv', 'w') as f:
        f.write("\t".join(parse_page(open('data/page1/place_0.html'), 'url').keys())+'\n')
        for i in range(1, end):
            f.write(open(f'refined_data/place_{i}.tsv', 'r').read()+'\n')
    return end

def generate_queries(domain):
    docs = domain.apply(lambda d: [word.lower() for word in wd_tokens.tokenize(d) if word not in stop_words])
    queries = []
    for doc in docs:
        for _ in range(3):
            queries.append(  " ".join(random.sample(doc, np.random.randint(2,4))) )
    return queries
