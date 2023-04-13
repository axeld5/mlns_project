#Import libraries for different Graph Neural Network
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import torch
import torch_geometric
from torch_geometric.utils.convert import to_networkx
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx

from gnn_models.gat import GAT 
from gnn_models.gcn import GCN 
from gnn_models.graphsage import GraphSAGE
from train_test_models import train, test


if __name__ == "__main__":    
    #set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the PubMed dataset
    dataset = Planetoid(root='.', name="Pubmed")
    data = dataset[0]

    # view the dataset details
    # Print information about the dataset
    print(f'Dataset: {dataset}')
    print('-------------------')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of nodes: {data.x.shape[0]}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Print information about the graph
    print(f'\nGraph:')
    print('------')
    print(f'Training nodes: {sum(data.train_mask).item()}')
    print(f'Evaluation nodes: {sum(data.val_mask).item()}')
    print(f'Test nodes: {sum(data.test_mask).item()}')
    print(f'Edges are directed: {data.is_directed()}')
    print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Graph has loops: {data.has_self_loops()}')

    #Visualize the graph using networkx
    plt.figure(figsize=(10, 10))
    pubmed = torch_geometric.data.Data(x=data.x[:500], edge_index=data.edge_index[:500])
    g = torch_geometric.utils.to_networkx(pubmed, to_undirected=True)
    pubmedGraph = to_networkx(pubmed)
    node_labels = data.y[list(pubmedGraph.nodes)].numpy()
    """line commented because it doesn't work yet"""
    #nx.draw(g, cmap=plt.get_cmap('Set1'),node_color = node_labels,node_size=75,linewidths=6)
    
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
        input_nodes=data.train_mask,
    )

    num_node_features = dataset.num_features 
    num_classes = dataset.num_classes

    # Create GraphSAGE
    graphsage = GraphSAGE(num_node_features, 64, num_classes).to(device)
    print(graphsage)

    # Train GraphSAGE
    train(graphsage, dataset, 200, device)

    # Test GraphSAGE
    print(f'\nGraphSAGE test accuracy: {test(graphsage, data)*100:.2f}%\n')

    # Create GAT
    gat = GAT(num_node_features, num_classes).to(device)
    print(gat)

    # Train Graph Attention Network
    train(gat, dataset, 200, device)

    # Test GAT
    print(f'\nGraph Attention Network test accuracy: {test(gat, data)*100:.2f}%\n')

    # Create GCN
    gcn = GCN(num_node_features, num_classes).to(device)
    print(gcn)

    # Train GCN
    train(gcn, dataset, 200, device)

    # Test GCN
    print(f'\nGCN test accuracy: {test(gat, data)*100:.2f}%\n')