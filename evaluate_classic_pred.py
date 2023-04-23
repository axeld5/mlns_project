from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from load_datasets import load_dataset
from classic_methods.clustering.clust_classifier import get_community_based_pred
from classic_methods.node_features.feat_classifier import (
    FeatureExtractor,
    feature_prediction,
)
from classic_methods.node2vec.node2vec import (
    get_node2vec_embeddings,
    node2vec_prediction,
)
import networkx as nx
import csv
import pandas as pd
from tqdm import tqdm
import scipy
import numpy as np


def BERT_attrs(G, df_products_info):
    """
    Loads the BERT embeddings and labels from `df_products_info` dataframe and assign them to node attributes of the graph G.
    """
    n_attrs = {}
    for node in tqdm(G.nodes, desc="Adding node embedding and class to the graph"):
        # Fetch node information (BERT embedding of the title of the product and label (class))
        node_info = df_products_info[df_products_info.ASIN == node]
        BERT_emb = node_info.BERT_embeddings.values[0]
        # The BERT_emb is a string of the BERT embedding. Thus the float values need to be retrieved.
        BERT_emb = BERT_emb.split("[")[-1].split("]")[0].split(" ")
        corrected_BERT_emb = []
        for elt in BERT_emb:
            if elt != "":
                if "\n" not in elt:
                    corrected_BERT_emb.append(float(elt))
                else:
                    elt = elt[:-2]
                    corrected_BERT_emb.append(float(elt))
        n_attrs[node] = {"x": corrected_BERT_emb, "y": int(node_info.Group_int)}
    return n_attrs


def TF_IDF_attrs(G, df_products_info):
    """
    Loads the TF-IDF embeddings and labels from `df_products_info` dataframe and assign them to node attributes of the graph G.
    """
    n_attrs = {}
    for node in tqdm(G.nodes, desc="Adding node embedding and class to the graph"):
        # Fetch node information (TF-IDF embedding of the title of the product and label (class))
        node_info = df_products_info[df_products_info.ASIN == node]
        # TF-IDF embedding are stored as string of sparse scipy matrix
        TFIDF_emb = node_info.TF_IDF.values[0]
        try:
            data = [
                np.short(float(elt.split("\t")[-1][:-2]))
                for elt in TFIDF_emb.split("\n")
            ]
            pos = [
                int(elt.split(")\t")[0].split(", ")[-1])
                for elt in TFIDF_emb.split("\n")
            ]
            TFIDF_emb = scipy.sparse.csr_matrix(
                (data, ([0] * len(pos), pos)), shape=(1, 886), dtype=np.short
            ).toarray()
            TFIDF_emb = np.squeeze(TFIDF_emb)
        except AttributeError:
            TFIDF_emb = np.zeros((886))
        n_attrs[node] = {"x": TFIDF_emb, "y": int(node_info.Group_int)}
    return n_attrs


def import_amazon_dataset(edge_path, embeddings_path, n_nodes=8579, embedding="BERT"):
    G = nx.Graph()
    csvreader = csv.reader(open(edge_path))
    # Load the 8579 first nodes (= average number of nodes in CiteSeer (3313), Cora (2708) and Pubmed (19717))
    for row in csvreader:
        if row and row[0] and row[1]:
            G.add_edge(row[0], row[1])
        if len(G.nodes) > n_nodes:
            break
    df_products_info = pd.read_csv(embeddings_path)
    # Add the embeddings to the (correct) node features
    if embedding == "BERT":
        n_attrs = BERT_attrs(G, df_products_info)
    else:
        n_attrs = TF_IDF_attrs(G, df_products_info)
    nx.set_node_attributes(G, n_attrs)
    # Reindexing G node from ASIN (Amazon product id) to [0, 1, ..., n_nodes-1]
    G = nx.convert_node_labels_to_integers(
        G, first_label=0, ordering="default", label_attribute="ASIN"
    )
    return G


if __name__ == "__main__":
    # Uncomment one of the following line in order to load to load either Pubmed, Citeseer or Cora or the Amazon dataset using BERT embedding or TFI-IDF.
    g = load_dataset("Citeseer", to_netx=True)  # "Pubmed", "Cora", "Citeseer"
    # g = import_amazon_dataset('edges.csv', 'BERT_embeddings2.csv', embedding='BERT')
    # g = import_amazon_dataset('edges.csv', 'TF_IDF_embeddings.csv', embedding='TF_IDF')
    
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
