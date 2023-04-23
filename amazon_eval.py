#Import libraries for different Graph Neural Network
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

from load_datasets import load_dataset

from gnn_models.gat import GAT 
from gnn_models.gcn import GCN 
from gnn_models.graphsage import GraphSAGE
from train_test_models import train, test

import numpy as np
import networkx as nx
from evaluate_classic_pred import import_amazon_dataset


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


if __name__ == "__main__":    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Uncomment one of the following line in order to load to load either Pubmed, Citeseer or Cora or the Amazon dataset using BERT embedding or TFI-IDF.
    nx_graph = import_amazon_dataset(edge_path='amazon_dataset/edges.csv', embeddings_path='amazon_dataset/BERT_embeddings2.csv', embedding="BERT")

    # Print information about the dataset
    print(f'Dataset: {nx_graph}')
    print('-------------------')
    print(f'Number of graphs: {1}')
    print(f'Number of nodes: {len(nx_graph.nodes)}')
    print(f'Number of edges: {len(nx_graph.edges)}')

    print(f'Number of features: {len([elt[1] for elt in nx_graph.nodes.data(data="x")][0])}')
    print(f'Number of classes: {len(set(nx.get_node_attributes(nx_graph, "y").values()))}')
    # Retrieve node embeddings, labels and edges
    node_embeddings = np.array([elt[1] for elt in nx_graph.nodes.data(data='x')])
    labels = np.array([elt[1] for elt in nx_graph.nodes.data(data='y')])
    edge_index = np.array([[int(line.split(' ')[:-1][0]), int(line.split(' ')[:-1][1])] for line in nx.generate_edgelist(nx_graph)])

    # Convert numpy array to tensors
    edge_index, node_embeddings, labels = torch.from_numpy(edge_index.reshape(2, -1)), torch.from_numpy(node_embeddings), torch.from_numpy(labels)
    # Convert to correct types
    labels, edge_index, node_embeddings = labels.type(torch.LongTensor), edge_index.type(torch.LongTensor), node_embeddings.type(torch.float)

    data = Data(x=node_embeddings,
                y=labels,
                edge_index = edge_index)
    
    # Split the data for training
    train_mask, test_mask = train_test_masks(data)

    # Print information about the graph
    print(f'\nGraph:')
    print('------')
    print(f'Training nodes: {sum(train_mask).item()}')
    print(f'Evaluation nodes: {sum(test_mask).item()}')
    print(f'Test nodes: {sum(test_mask).item()}')
    print(f'Edges are directed: {data.is_directed()}')
    print(f'Graph has loops: {data.has_self_loops()}')

    # Add additional arguments to `data`:
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    
    data = data.to(device)

    # Create batches with neighbor sampling
    #A NeighborLoader is a data loader that performs neighbor sampling for GNN's
    #Allows for mini-batch training of GNNs on large-scale graphs where full-batch training is not feasible.
    #num_neighbors denotes how many neighbors are sampled for each node in each iteration.
    #https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader
    train_loader = NeighborLoader(
        data,
        num_neighbors=[5, 10],
        batch_size=16,
        input_nodes=train_mask
    )

    num_node_features = data.num_features 
    num_classes = len(set(nx.get_node_attributes(nx_graph, "y").values())) 

    # Create GraphSAGE
    graphsage = GraphSAGE(num_node_features, 64, num_classes).to(device)
    print(graphsage)

    # Train GraphSAGE
    train(graphsage, train_loader, 200, device)

    # Test GraphSAGE
    print(f'\nGraphSAGE test accuracy: {test(graphsage, data)*100:.2f}%\n')

    # Create GAT
    gat = GAT(num_node_features, num_classes).to(device)
    print(gat)

    # Train Graph Attention Network
    train(gat, train_loader, 200, device)

    # Test GAT
    print(f'\nGraph Attention Network test accuracy: {test(gat, data)*100:.2f}%\n')

    # Create GCN
    gcn = GCN(num_node_features, num_classes).to(device)
    print(gcn)

    # Train GCN
    train(gcn, train_loader, 200, device)

    # Test GCN
    print(f'\nGCN test accuracy: {test(gat, data)*100:.2f}%\n')