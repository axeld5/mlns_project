import networkx as nx

from torch_geometric.datasets import Planetoid
from torch_geometric.utils.convert import to_networkx

def load_dataset(dataset_name:str, to_netx:bool):
    #can be Pubmed, Cora or Citeseer
    dataset = Planetoid(root='.', name=dataset_name)
    data = dataset[0]
    if to_netx: 
        g = to_networkx(data, node_attrs=["x", "y"], to_undirected=True)
        return g 
    return data