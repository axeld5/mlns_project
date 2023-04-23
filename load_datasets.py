import networkx as nx
import csv
import pandas as pd
import scipy
import numpy as np
import torch


from tqdm import tqdm
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.convert import to_networkx
from torch_geometric.loader import NeighborLoader

def load_dataset(dataset_name:str, to_netx:bool):
    #can be Pubmed, Cora or Citeseer
    assert dataset_name.lower() in ("pubmed", "cora", "citeseer")
    dataset = Planetoid(root='.', name=dataset_name)
    data = dataset[0]
    if to_netx: 
        g = to_networkx(data, node_attrs=["x", "y"], to_undirected=True)
        return g 
    else: 
        train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask,
        )
        return train_loader, dataset
    return 
    


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

def train_test_masks(G):
    # Split the data 
    train_ratio = 0.75
    num_nodes = G.x.shape[0]
    num_train = int(num_nodes * train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_mask = torch.full_like(G.y, False, dtype=bool)
    train_mask[idx[:num_train]] = True
    test_mask = torch.full_like(G.y, False, dtype=bool)
    test_mask[idx[num_train:]] = True

    return train_mask, test_mask