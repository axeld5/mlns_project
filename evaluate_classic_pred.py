from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from load_datasets import load_dataset 
from classic_methods.clustering.clust_classifier import get_community_based_pred
from classic_methods.node_features.feat_classifier import FeatureExtractor, feature_prediction
from classic_methods.node2vec.node2vec import get_node2vec_embeddings, node2vec_prediction


if __name__ == "__main__":
    g = load_dataset("Citeseer", to_netx=True)
    node_list = list(g.nodes)
    label_list = [g.nodes[i]["y"] for i in range(len(node_list))]

    extractor = FeatureExtractor()
    features = extractor.global_feature_extract(g, node_list)
    train_features, test_features, train_labels, test_labels = train_test_split(features, label_list)
    comm_pred = feature_prediction(train_features, train_labels, test_features)
    print("global features based classifier f1 score =", f1_score(test_labels, comm_pred, average="weighted"))

    features = extractor.node_feature_extract(g, node_list)
    train_features, test_features, train_labels, test_labels = train_test_split(features, label_list)
    comm_pred = feature_prediction(train_features, train_labels, test_features)
    print("node features based classifier f1 score =", f1_score(test_labels, comm_pred, average="weighted"))

    features = extractor.full_feature_extract(g, node_list)
    train_features, test_features, train_labels, test_labels = train_test_split(features, label_list)
    comm_pred = feature_prediction(train_features, train_labels, test_features)
    print("node and global features based classifier f1 score =", f1_score(test_labels, comm_pred, average="weighted"))

    train_nodes, test_nodes, train_labels, test_labels = train_test_split(node_list, label_list)
    comm_pred = get_community_based_pred(g, train_nodes, train_labels, test_nodes)
    print("community based classifier f1 score =", f1_score(test_labels, comm_pred, average="weighted"))

    embeddings = get_node2vec_embeddings(g) 
    train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(embeddings, label_list)
    node2pred = node2vec_prediction(train_embeddings, train_labels, test_embeddings)
    print("node2vec embeddings based classifier f1 score =", f1_score(test_labels, node2pred, average="weighted"))