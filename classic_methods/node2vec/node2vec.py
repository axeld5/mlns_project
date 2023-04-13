import networkx as nx 

from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression

def get_node2vec_embeddings(g):
    model = Node2Vec(g, workers=8, p=1, q=1).fit(window=6, min_count=1, batch_words=4) 
    embeddings = model.wv.vectors
    return embeddings

def node2vec_prediction(train_embeddings, train_labels, test_embeddings): 
    clf = LogisticRegression()
    clf.fit(train_embeddings, train_labels)
    y_pred = clf.predict(test_embeddings) 
    return y_pred