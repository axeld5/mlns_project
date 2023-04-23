import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from load_datasets import load_dataset, import_amazon_dataset
from classic_methods.clustering.clust_classifier import get_community_based_pred
from classic_methods.node_features.feat_classifier import (
    FeatureExtractor,
    feature_prediction,
)
from classic_methods.node2vec.node2vec import (
    get_node2vec_embeddings,
    node2vec_prediction,
)


if __name__ == "__main__":
    # Uncomment one of the following line in order to load to load either Pubmed, Citeseer or Cora or the Amazon dataset using BERT embedding or TFI-IDF.
    g = load_dataset("Citeseer", to_netx=True)  # "Pubmed", "Cora", "Citeseer"
    # g = import_amazon_dataset('amazon_dataset/edges.csv', 'amazon_dataset/BERT_embeddings2.csv', embedding='BERT')
    # g = import_amazon_dataset('amazon_dataset/edges.csv', 'amazon_dataset/TF_IDF_embeddings.csv', embedding='TF_IDF')
    
    node_list = list(g.nodes)
    label_list = [g.nodes[i]["y"] for i in range(len(node_list))]

    extractor = FeatureExtractor()
    features = extractor.global_feature_extract(g, node_list)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, label_list, shuffle=False
    )
    comm_pred = feature_prediction(train_features, train_labels, test_features)
    print(
        "\n\nglobal features based classifier accuracy =",
        accuracy_score(test_labels, comm_pred),
        "\n\n",
    )
    print(
        "\n\nglobal features based classifier F1-score =",
        f1_score(test_labels, comm_pred, average="weighted"),
        "\n\n",
    )

    del features
    del train_features
    del test_features
    del train_labels
    del test_labels

    features = extractor.node_feature_extract(g, node_list)
    features = np.array(features, dtype=np.float16)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, label_list, shuffle=False
    )
    comm_pred = feature_prediction(train_features, train_labels, test_features)
    print(
        "\n\nnode features based classifier accuracy =",
        accuracy_score(test_labels, comm_pred),
        "\n\n",
    )
    print(
        "\n\nnode features based classifier F1-score =",
        f1_score(test_labels, comm_pred, average="weighted"),
        "\n\n",
    )

    features = extractor.full_feature_extract(g, node_list)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, label_list, shuffle=False
    )
    comm_pred = feature_prediction(train_features, train_labels, test_features)
    print(
        "\n\nnode and global features based classifier accuracy =",
        accuracy_score(test_labels, comm_pred),
        "\n\n",
    )
    print(
        "\n\nnode and global features based classifier F1-score =",
        f1_score(test_labels, comm_pred, average="weighted"),
        "\n\n",
    )

    train_nodes, test_nodes, train_labels, test_labels = train_test_split(
        node_list, label_list, shuffle=False
    )
    comm_pred = get_community_based_pred(g, train_nodes, train_labels, test_nodes)
    print(
        "\n\ncommunity based classifier accuracy =",
        accuracy_score(test_labels, comm_pred),
        "\n\n",
    )
    print(
        "\n\ncommunity based classifier F1-score =",
        f1_score(test_labels, comm_pred, average="weighted"),
        "\n\n",
    )

    embeddings = get_node2vec_embeddings(g)
    train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
        embeddings, label_list, shuffle=False
    )
    node2pred = node2vec_prediction(train_embeddings, train_labels, test_embeddings)
    print(
        "\n\nnode2vec embeddings based classifier accuracy =",
        accuracy_score(test_labels, node2pred),
        "\n\n",
    )
    print(
        "\n\nnode2vec embeddings based classifier F1-score =",
        f1_score(test_labels, node2pred, average="weighted"),
        "\n\n",
    )
