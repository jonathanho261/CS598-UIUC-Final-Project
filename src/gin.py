import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.train import train
from src.utils import read_data, SAMPLE_DATA_FILE_PATH

# From: https://github.com/zhchs/Disease-Prediction-via-GCN
class MLP(nn.Module):
    '''
    Multi-Layer Perceptron
    '''
    def __init__(self, in_channels, out_channels, embedding_dim, num_layers):
        super(MLP, self).__init__()

        self.num_layers = num_layers

        self.linears = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        if self.num_layers == 1:
            self.linears.append(nn.Linear(in_channels, out_channels))
        else:
            self.linears.append(nn.Linear(in_channels, out_channels))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(out_channels, out_channels))
            self.linears.append(nn.Linear(out_channels, embedding_dim))

        for layer in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d((out_channels)))

    def forward(self, x):
        h = x
        for layer in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
        return self.linears[self.num_layers - 1](h)

class GIN(nn.Module):
    '''
    Graph Isomorphic Module
    '''
    def __init__(self, in_channels, out_channels, embedding_dim, num_layers, num_sample=5):
        super(GIN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.num_layers = 2
        self.num_sample = num_sample
        
        self.embedding = nn.Embedding(in_channels, in_channels)
        self.layers = nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(2):
            if layer == 0:
                self.layers.append(MLP(self.in_channels, self.out_channels, self.out_channels, num_layers))
            else:
                self.layers.append(MLP(self.out_channels, self.out_channels, self.out_channels, num_layers))
            self.batch_norms.append(nn.BatchNorm1d(self.out_channels))
        self.layers.append(MLP(self.out_channels, self.out_channels, self.embedding_dim, num_layers))
        self.batch_norms.append(nn.BatchNorm1d(self.embedding_dim))
        
    def forward(self, nodes, adj_list):
        embedding = self.embedding(torch.LongTensor(nodes))
        aggregated_embedding = self.aggregate(nodes, adj_list)

        h = torch.add(embedding, aggregated_embedding)
        for layer in range(self.num_layers):
            pooled_rep = self.layers[layer](h)
            h = self.batch_norms[layer](pooled_rep)
            h = F.relu(h)
        combined = h.t()
        return combined

    def aggregate(self, nodes, adj_list):
        sampled_neighbors = []
        unique_neighbors = set()
        for node in nodes:
            neighbors = set()
            if len(adj_list[node]) >= self.num_sample:
                neighbors = set(random.sample([*adj_list[node]], self.num_sample))
            else:
                neighbors = adj_list[node]
            neighbors.add(node)
            sampled_neighbors.append(neighbors)
            unique_neighbors = unique_neighbors.union(neighbors)

        column_indices = [i
                          for neighbor in sampled_neighbors for i, n in enumerate(neighbor)]
        row_indices = [i for i in range(len(sampled_neighbors)) for j in range(len(sampled_neighbors[i]))]

        embedding = self.embedding(torch.LongTensor(list(unique_neighbors)))

        mask = torch.zeros(len(sampled_neighbors), len(unique_neighbors))
        mask[row_indices, column_indices] = 1

        aggregate_feats = mask.mm(embedding)
        return aggregate_feats

class GinClassifier(nn.Module):

    def __init__(self, emb_size, num_classes):
        super(GinClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(emb_size, num_classes),  
            nn.ReLU(),
        )
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
                                
    def forward(self, embeds):
        embeds = torch.swapaxes(embeds, 1, 0)
        return self.layer(embeds)


if __name__ == '__main__':
    patients, adj_list, rare_diseases, labels, node_map, train_nodes, test_nodes = read_data(SAMPLE_DATA_FILE_PATH)
    
    in_channels = 10000
    out_channels = 512
    embedding_dim = 1000
    num_layers = 5

    gin = GIN(
        in_channels=in_channels,
        out_channels=out_channels,
        embedding_dim=embedding_dim,
        num_layers=num_layers
    )

    classifer = GinClassifier(emb_size=out_channels, num_classes=labels.shape[1])

    train(gin, classifer, labels, adj_list, train_nodes)


