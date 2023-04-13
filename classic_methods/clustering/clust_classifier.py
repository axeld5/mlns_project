import numpy as np

from networkx.algorithms.community import greedy_modularity_communities

def get_community_based_pred(g, train_nodes, train_labels, test_nodes):
    print("starting establishing communities")
    c = greedy_modularity_communities(g)
    print("done establishing communities")
    comm_dict = get_community_dict(g, c)
    comm_lab_dict = get_majority_label(train_nodes, train_labels, c)
    comm_pred = get_pred(comm_dict, comm_lab_dict, test_nodes)
    return comm_pred

def get_community_dict(g, c):
    comm_dict = {}
    for i, community in enumerate(c):  
        for node in g.nodes: 
            if node in community:
                comm_dict[node] = i
    return comm_dict

def get_majority_label(train_nodes, train_labels, c):
    comm_lab_dict = {}
    for i, community in enumerate(c):
        comm_lab_dict[i] = []
        for j, node in enumerate(train_nodes):
            if node in community:
                comm_lab_dict[i].append(train_labels[j]) 
        comm_lab_dict[i] = max(comm_lab_dict[i], key=comm_lab_dict[i].count)
    return comm_lab_dict

def get_pred(comm_dict, comm_lab_dict, test_nodes):
    comm_pred = np.zeros(len(test_nodes))
    for i, node in enumerate(test_nodes):
        comm_idx = comm_dict[node]
        comm_pred[i] = comm_lab_dict[comm_idx]
    return comm_pred