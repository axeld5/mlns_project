import tqdm
import networkx as nx 
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

class FeatureExtractor:
    def __init__(self) -> None:
        pass 
    
    def global_feature_extract(self, graph, samples):
        feature_vector = [] 
        deg_centrality = nx.degree_centrality(graph)
        print("degree centrality computed")
        betweeness_centrality = nx.betweenness_centrality(graph, k=500)
        print("betweeness centrality computed")
        p_rank = nx.pagerank(graph)
        print("pagerank computed")
        hits = nx.hits(graph)
        print("hits computed")
        hubs = hits[0]
        auth = hits[1]
        for node in tqdm.tqdm(samples):
            dg_ctrl_node = deg_centrality[node]
            btw_ctrl_node = betweeness_centrality[node]
            hubs_node = hubs[node]
            auth_node = auth[node]
            p_rank_node = p_rank[node]
            node_features = np.array([dg_ctrl_node, btw_ctrl_node, hubs_node, auth_node, p_rank_node])
            feature_vector.append(node_features) 
        feature_vector = np.array(feature_vector)
        return feature_vector 
    
    def node_feature_extract(self, graph, samples):
        feature_vector = [] 
        for node in tqdm.tqdm(samples):
            node_embeddings = graph.nodes[node]["x"]
            feature_vector.append(node_embeddings) 
        feature_vector = np.array(feature_vector, dtype=np.float16)
        return feature_vector 

    def full_feature_extract(self, graph, samples):
        global_feat_vector = self.global_feature_extract(graph, samples) 
        node_feat_vector = self.node_feature_extract(graph, samples) 
        try:
            feature_vector = np.concatenate([global_feat_vector, node_feat_vector], axis=1)
        except ValueError:
            node_feat_vector = np.squeeze(node_feat_vector)
            feature_vector = np.concatenate([global_feat_vector, node_feat_vector], axis=1)
        return feature_vector

def feature_prediction(train_features, train_labels, test_features): 
    clf = LogisticRegression(max_iter=1000)
    train_features = np.squeeze(train_features)
    clf.fit(train_features, train_labels)
    test_features = np.squeeze(test_features)
    y_pred = clf.predict(test_features) 
    return y_pred