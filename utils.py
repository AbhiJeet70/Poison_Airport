import torch
import numpy as np
import random
from torch_geometric.datasets import Airports
from torch_geometric.transforms import NormalizeFeatures

def load_airports_data(country):
    dataset = Airports(root='/tmp/Airports', name=country, transform=NormalizeFeatures())
    data = dataset[0]
    return data

def split_indices(num_nodes, train_ratio=0.7, val_ratio=0.1):
    indices = np.random.permutation(num_nodes)
    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)
    train_idx = torch.tensor(indices[:train_end], dtype=torch.long)
    val_idx = torch.tensor(indices[train_end:val_end], dtype=torch.long)
    test_idx = torch.tensor(indices[val_end:], dtype=torch.long)
    return train_idx, val_idx, test_idx

def print_dataset_statistics(data, dataset_name):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    num_features = data.num_node_features
    num_classes = data.y.max().item() + 1

    print(f"Dataset: {dataset_name}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")

def evenly_perturb_edges(data, perturbation_percentage):
    num_edges_to_perturb = int(data.num_edges * perturbation_percentage)
    edge_index = data.edge_index.cpu().numpy()
    perm = np.random.permutation(data.num_edges)
    perturbed_edge_index = edge_index[:, perm[num_edges_to_perturb:]]
    data.edge_index = torch.tensor(perturbed_edge_index, dtype=torch.long)
    return data

def concentrated_perturb_edges(data, perturbation_percentage):
    degree = data.edge_index[0].bincount()
    num_nodes_to_perturb = int(len(degree) * perturbation_percentage)
    top_k_nodes = torch.topk(degree, num_nodes_to_perturb).indices

    mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
    for node in top_k_nodes:
        mask &= (data.edge_index[0] != node) & (data.edge_index[1] != node)

    data.edge_index = data.edge_index[:, mask]
    return data, top_k_nodes
