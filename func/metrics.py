import pandas as pd

def precision_k(y, prediction, k=None):
    '''How many relevant items are in the top k results'''
    if k is None:
        k = len(y)
    prediction = prediction[:k]
    return len(set(y).intersection(set(prediction))) / k

def recall_k(y, pred, k=15):
    '''How many relevant items are in our top k results over all the relevant items for that query'''
    relevant = set(y).intersection(set(pred))
    relevant_topk = set(pred[:k]).intersection(set(y))
    return len(relevant_topk) / len(relevant) if len(relevant) > 0 else 0

def topk_accuracy(y,pred, k=None):
    '''Is at least one of the relevent items in the top k results'''
    return precision_k(y, pred, k=k) > 0

def mean_average_precision(y,pred):
    m_ap = 0
    for k in range(1,len(y)):
        m_ap+=precision_k(y, pred, k)
    return m_ap/len(y)

def k_relevance(y,preds, k=5):
    ''' How far do we have to go before finding the first k relevant items? '''
    y = y[:k]
    rel = 0
    for r in y:
        if r not in preds:
            return len(preds)
        rel = max(rel, preds.index(r))
    return rel

def score_metrics(query, y, engine):
    '''combines all different metrics'''
    y = [y_i for y_i in y if y_i != 'NA']
    if len(y) == 0:
        return None
    pred = engine.search(query, k=1000).placeUrl.tolist()
    res = dict()
    res['Top-k Accuracy'] = topk_accuracy(y, pred)
    res['Precision@k'] = precision_k(y, pred)
    res['Mean Average Precision'] = mean_average_precision(y, pred)
    res['Recall@k'] = recall_k(y, pred)
    if res['Top-k Accuracy']:
        res['K-Relevance'] = k_relevance(y, pred, k=int(res['Precision@k']*len(y)))
    else:
        res['K-Relevance'] = 0
    return res

def evaluate_engine(queries, engine, score=score_metrics):
    scores = None

    for _, (query, *y) in queries.iterrows():
        res = score(query, y, engine)
        if res is None:
            continue
        if scores is None:
            scores = res
            continue
        for metric, val in res.items():
            scores[metric] += val
    scores = {m:v/queries.shape[0] for m,v in scores.items()}
        
    print("\n".join([f'{k:30} {round(v,3)}' for k,v in scores.items()]))