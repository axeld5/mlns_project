import tqdm
import networkx as nx 
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

class FeatureExtractor:
    def __init__(self, scaler=MinMaxScaler()) -> None:
        self.scaler = scaler

    def feature_extract(self, graph, samples):
        """
        Creates a feature vector for each edge of the graph contained in samples 
        """
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
            
            ## Global FEATURES
            # Degree Centrality
            dg_ctrl_node = deg_centrality[node]
            
            btw_ctrl_node = betweeness_centrality[node]

            #Auth
            hubs_node = hubs[node]
            auth_node = auth[node]

            #prank
            p_rank_node = p_rank[node]

            ## NODE EMBEDDINGS 
            #They carry the most information
            node_embeddings = graph.nodes[node]["x"]
            node_features = np.array([dg_ctrl_node, btw_ctrl_node, hubs_node, auth_node, p_rank_node])
            
            features = np.concatenate([node_embeddings, node_features])
            feature_vector.append(features) 
        feature_vector = np.array(feature_vector)
        feature_vector = self.scaler.fit_transform(feature_vector)
        return feature_vector

def feature_prediction(train_features, train_labels, test_features): 
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_features, train_labels)
    y_pred = clf.predict(test_features) 
    return y_pred