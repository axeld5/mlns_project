import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

'''
Graph SAGE: SAmpling and aggreGatE, 
Samples only a subset of neighboring nodes at different depth layers, 
and then the aggregator takes neighbors of the previous layers and aggregates them
'''
class GraphSAGE(torch.nn.Module):
  """GraphSAGE"""
  def __init__(self, num_node_features, hidden_dim, num_classes):
    super().__init__()
    self.sage1 = SAGEConv(num_node_features, hidden_dim*2)
    self.sage2 = SAGEConv(hidden_dim*2, hidden_dim)
    self.sage3 = SAGEConv(hidden_dim, num_classes)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.01,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    h = self.sage1(x, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.sage2(h, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.2, training=self.training)
    h = self.sage3(h, edge_index)
    return h, F.log_softmax(h, dim=1)