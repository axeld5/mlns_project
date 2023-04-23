import argparse
import torch
import numpy as np 
import networkx as nx 

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

from gnn_models.gat import GAT
from gnn_models.gcn import GCN
from gnn_models.graphsage import GraphSAGE
from load_datasets import load_dataset, import_amazon_dataset, train_test_masks

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def train(model, train_loader):
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    if isinstance(model, GAT):
       nb_epochs = 400
       optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    elif isinstance(model, GCN):
       nb_epochs = 300
       optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    elif isinstance(model, GraphSAGE):
        nb_epochs = 300
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(nb_epochs+1):
      total_loss = 0
      acc = 0
      val_loss = 0
      val_acc = 0

      # Train on batches
      for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        
        _, out = model(batch.x, batch.edge_index)
        loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
        total_loss += loss
        acc += accuracy(out[batch.train_mask].argmax(dim=1), 
                        batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
        val_acc += accuracy(out[batch.val_mask].argmax(dim=1), 
                                batch.y[batch.val_mask])

      # Print metrics every 10 epochs
      if(epoch % 10 == 0):
          print(f'Epoch {epoch:>3} | Train Loss: {total_loss/len(train_loader):.3f} '
                f'| Train Acc: {acc/len(train_loader)*100:>6.2f}% | Val Loss: '
                f'{val_loss/len(train_loader):.2f} | Val Acc: '
                f'{val_acc/len(train_loader)*100:.2f}%')

def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launches GNN training")
    parser.add_argument("--model", type=str, default="rbf", help="Kernel to use.")
    parser.add_argument("--dataset", type=str, default="rbf", help="Kernel to use.")
    args = parser.parse_args()

    train_loader, dataset = load_dataset(args.dataset, to_netx=False)
    data = dataset[0]
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    if args.model.lower() == "gcn":
       model = GCN(num_features, num_classes)
    elif args.model.lower() == "gat":
       model = GAT(num_features, num_classes)
    elif args.model.lower() == "graphsage":
       model = GraphSAGE(num_features, 64, num_classes)
    else:
       raise NotImplementedError

    # Launches training
    train(model=model, train_loader=train_loader)

    # Launches test evaluation
    acc = test(model=model, data=data)
    print(f"Final accuracy for {args.model} on {args.dataset}: {acc}")