import pandas as pd
import numpy as np
import hashlib
import heapq as hq
from datetime import datetime
from scipy.sparse import dok_matrix
from collections import Counter, defaultdict
import math


from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

wd_tokens = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

class Engine:
    '''Class that represents a Simple Search Engine'''
    SIZE = 2**63 - 1 # max size of the hash
    def __init__(self, domain:pd.DataFrame, out_columns:list):
        self.domain = domain
        self.corpus = None
        self.out_columns = out_columns
        self.matrix = dok_matrix((Engine.SIZE,domain.index.max()+1), dtype=np.float32) 

    def fit(self, corpus_columns:list):
        '''fits the engine to a corpus of documents, by creating a term indicator matrix'''
        self.corpus = self.domain[corpus_columns]
        for _,search_field in self.corpus.items():
            search_field.to_frame()\
                .apply( 
                    lambda doc: self.document_process(doc.name, doc.iloc[0]),
                    axis=1)
        return self
    
    def document_process(self, doc_id, doc):
        '''Processes each document'''
        for term in Engine.get_terms(doc):
            self.add_term(term, doc_id) 

    def add_term(self, term, doc, value=1):
        self.matrix[term, doc] += value
        
    def search(self, query, k=15):
        '''Returns all the documents that contain all the terms in the query'''
        terms = Engine.get_terms(query)
        docs_score = self.matrix[terms].sum(axis=0).A1
        docs = np.where(docs_score == len(terms))[0]
        return self.domain[self.out_columns].iloc[docs].iloc[:k]

    @staticmethod
    def clean_str(s, hash=False, tokenizer = wd_tokens.tokenize, stemmer = ps.stem):
        '''Cleans, stems, and eventually hashes a query string into a list of terms'''
        s = s.replace('\n', '')
        terms = [stemmer(word.lower()) for word in tokenizer(s) if word not in stop_words]
        if hash:
            return [Engine.hash_term(term) for term in terms]
        return terms

    @staticmethod
    def get_terms(q, hash=True):
        '''Returns the terms in the query without repetitions'''
        return list(set(Engine.clean_str(q, hash)))
   
    @staticmethod
    def hash_term(s):
        '''Hash of a string in int, with max size SIZE'''
        return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (Engine.SIZE)
    
class SimilarityEngine(Engine):
    '''This class represents an improved version of the orginal SimpleEngine, that obtains documents by similarity'''
    IDF = lambda N, df: np.log(N / df) + 1

    def __init__(self, domain:pd.DataFrame, out_columns:list):
        super().__init__(domain, out_columns)
        self.idf = None
        self.norm = dict()

    def document_process(self, doc_id, doc):
        v_doc = SimilarityEngine.term_frequency(doc)
        for term, tf in v_doc.items():
            self.add_term(term, doc_id, value=tf)

    def fit(self, corpus_columns:list):
        super().fit(corpus_columns)

        # Inverse Document Frequency
        self.idf = Counter([ i for i,_ in self.matrix.keys() ])
        for t, fr in self.idf.items():
            self.idf[t] = SimilarityEngine.IDF(self.matrix.shape[1], fr)

        for i,j in self.matrix.keys():
            self.matrix[i,j] *= self.idf[i]

        csc = self.matrix.tocsc()
        for doc_id in self.domain.index:
            self.norm[doc_id] = np.sqrt(np.sum([v**2 for v in csc.getcol(doc_id).data]))
        return self

    def retrieve_doc_vec(self, terms):
        '''Retrieves vector rappresentation of documents given some terms'''
        doc_vec = {}
        m = self.matrix[sorted(terms.keys()),:].tocsc()
        for i in m.nonzero()[1]:
            doc_vec[i] = m.getcol(i).toarray()[:,0]
        return doc_vec

    def search(self, query, k=10, all_columns=False):
        '''returns the k most similar documents to the query'''
        docs_score, _, _ = self.raw_search(query, k)
        docs = [doc_id for _, doc_id in hq.nlargest(k, docs_score)]

        res = self.domain.iloc[docs].copy()
        res['similarity'] = [sim for sim, _ in hq.nlargest(k, docs_score)]
        if all_columns:
            return res[res.similarity != 0]
        return res[self.out_columns + ['similarity']][res.similarity != 0]
    
    def similar_vectors(self, query, k=10):
        '''Returns the k most similar vectors to the query'''
        docs_score, query_vec, doc_vec = self.raw_search(query, k)
        vecs = {self.domain.placeName.iloc[id]:doc_vec[id]/np.linalg.norm(doc_vec[id]) for _, id in hq.nlargest(k, docs_score)}
        vecs['QUERY'] = query_vec / np.linalg.norm(query_vec)
        return vecs

    def raw_search(self, query, k=7200):
        '''Returns an heap with the similarity scores for the query'''
        terms = SimilarityEngine.term_frequency(query)
        query_vec = np.array([self.idf[t] * tf for t, tf in terms.items() ])

        doc_vec = self.retrieve_doc_vec(terms)
        
        docs_score = []
        for doc_id, d in doc_vec.items():
            similarity = SimilarityEngine.cosine_sim(d,query_vec)
            # keep only the first k values in the heap
            if k >= 0:
                k -= 1
                hq.heappush(docs_score, (similarity, doc_id))
            else:
                hq.heappushpop(docs_score, (similarity, doc_id))
        return docs_score, query_vec, doc_vec

    @staticmethod
    def term_frequency(d, hash=True): 
        '''Returns the term frequency of a document in term:value pairs'''
        terms = Engine.clean_str(d, hash)
        v = Counter(terms)
        return {t:cnt/sum(v.values()) for t, cnt in v.items()}

    @staticmethod
    def vectorize(s):
        '''returns an array of the term frequencies for a given query'''
        tf = SimilarityEngine.term_frequency(s)
        return np.array(list(tf.values()))

    @staticmethod
    def cosine_sim(v1, v2, norm1=None):
        '''computes the cosine similarity between vectors'''
        n1 = np.linalg.norm(v1) if norm1 is None else norm1
        n2 = np.linalg.norm(v2)
        return v1.dot(v2) / n1 / n2 if n1 * n2 != 0  else 0
    
class RankEngine(SimilarityEngine):
    '''Searches and then ranks'''
    def __init__(self, domain:pd.DataFrame, out_columns:list,
            weights = {
                'editors_score' : 2,
                'wants_score' : 1,
                'visited_score' : 1,
                'related_score' : 0
            },
            ratio = 0.5,
            lat='placeAlt',
            lng='placeLng',
            time='placePubDate'):
        super().__init__(domain, out_columns)
        self.ratio = ratio
        self.set_weights(weights)
        self.time_column = time
        self.lng_column = lng
        self.lat_column = lat

    def set_weights(self, weights):
        '''updates a weight value'''
        self.weights = weights
        norm = sum(self.weights.values())
        for col, w in self.weights.items():
            self.domain[col] = self.normalize(self.domain[col])
            self.weights[col] /= norm
        self.domain['importance'] = self.domain.apply(self.importance, axis=1)

    def importance(self,row):
        '''computes the rank score of the document'''
        score = self.weights.copy()
        for col in self.weights.keys():
            score[col] *= row[col]
        return sum(score.values())

    def location_search(self, query, lat, lng, k =10):
        '''Searches the query and ranks by distance from the coordinates'''
        docs_sim, _, _ = super().raw_search(query)
        distance = self.domain[[self.lat_column, self.lng_column]].apply(
            lambda r: RankEngine.haversine_distance(r[0], r[1], lat, lng), 
            axis=1)
        distance = 1 - RankEngine.normalize(distance)
        return self.present(docs_sim, distance, k, [self.lat_column,self.lng_column])

    def time_search(self, query, date=datetime.now(), k=10):
        '''Searches the query and ranks by distance from the given timestamp'''
        docs_sim, _, _ = super().raw_search(query)
        distance = np.abs((self.domain[self.time_column] - date).dt.total_seconds())
        distance = 1 - RankEngine.normalize(distance)
        return self.present(docs_sim, distance, k, [self.time_column])

    def search(self, query, k=10, columns = []):
        '''Searches the query and ranks by importance'''
        docs_sim, _, _ = super().raw_search(query)
        importance = self.domain.importance
        return self.present(docs_sim, importance, k , columns)

    def present(self, docs_sim, ranking, k, columns=[]):
        '''Takes an heap of similarity values as input and a ranking score and outputs a pandas dataframe'''
        docs_score = []
        for sim, doc_id in hq.nlargest(k*100, docs_sim):
            score = self.ratio*sim + (1-self.ratio) * ranking.iloc[doc_id]
            hq.heappush(docs_score, (score ,doc_id))
        
        docs = [doc_id for _, doc_id in hq.nlargest(k, docs_score)]
        res = self.domain.iloc[docs].copy()
        res['score'] = [score for score, _ in hq.nlargest(k, docs_score)]
        if columns == True:
            return res
        return res[self.out_columns + columns + ['score']].iloc[:k]

    @staticmethod
    def haversine_distance(lat_1, lng_1, lat_2, lng_2): 
        '''Calculates distance between coordinates'''
        # https://stackoverflow.com/a/44743104/9419492
        lng_1, lat_1, lng_2, lat_2 = map(np.radians, [lng_1, lat_1, lng_2, lat_2])
        d_lat = lat_2 - lat_1
        d_lng = lng_2 - lng_1 

        temp = math.sin(d_lat / 2) ** 2 + math.cos(lat_1) * math.cos(lat_2) * math.sin(d_lng / 2) ** 2
        return 6373.0 * (2 * math.atan2(math.sqrt(temp), math.sqrt(1 - temp)))

    @staticmethod
    def normalize(column):
        '''returns min-max normalization of an array or pandas series'''
        column -= column.min()
        return column / column.max()
    
class FilterEngine():
    '''Searches across multiple corpora of documents and then filters them '''
    def __init__(self, domain):
        self.domain = domain
        self.corpora = []         
        self.searches = dict()
        self.corpora_weights = dict()
        
    def fit(self, corpora):
        self.corpora = corpora
        for col in corpora:
            self.searches[col] = SimilarityEngine(self.domain,['placeName'])
            self.searches[col].fit([col])
            self.corpora_weights[col] = 1/np.sqrt(len(self.searches[col].idf))
        self.corpora_weights = {k:v/sum(self.corpora_weights.values()) for k,v in self.corpora_weights.items()}

    def filter(self, row, 
        editors = set(),
        tags = set(),
        min_visited = 0, max_visited = math.inf,
        lists = set()):
        '''returns True if all the constraints are respected'''
        rowEditors, rowTags, rowLists = map(lambda x: eval(x) if x is not np.nan else set(), [row.placeEditors, row.placeTags, row.placeRelatedLists])
        return editors.issubset(rowEditors)  \
            and tags.issubset(rowTags) \
            and lists.issubset(rowLists) \
            and row.numPeopleVisited > min_visited and row.numPeopleVisited < max_visited

    def similarities(self, queries):
        sim = pd.DataFrame()
        if len(queries) == 0:
            sim = sim.reindex(index=data.index)
            sim['similarity'] = 0
            return sim

        for col, query in queries.items():
            col_sim = self.searches[col].search(query, k=self.domain.shape[0])['similarity']
            col_sim.rename(col+'_sim', inplace=True)
            sim = pd.concat([sim,col_sim], axis='columns')
        sim['similarity'] = 0
        for col in queries.keys():
            sim['similarity'] = sim.similarity.add(self.corpora_weights[col]*sim[col+'_sim'], fill_value=0)
        return sim

    def search(
        self, queries = dict(),
        editors = [],
        tags = [],
        min_visited = 0, max_visited = math.inf,
        lists = [] ,
        columns=[],
        k = 15):
        '''Searches the across multiple corpuses and filters the results on the constraints'''
        sim = self.similarities({k:v for k,v in queries.items() if k in self.corpora})

        docs = pd.concat([ self.domain, sim ], axis='columns', join='inner')
        filtered = docs.apply(
            lambda r : self.filter(r, set(editors), set(tags), min_visited, max_visited, set(lists)),
            axis=1)
        res = docs[filtered].nlargest(k, 'similarity')
        if columns==True:
            return res
        return res[list(set(['placeName', 'placeUrl', 'similarity'] +list( queries.keys() )))]