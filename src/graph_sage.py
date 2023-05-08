import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda import memory_summary

import matplotlib.pyplot as plt

from src.model_helpers import train, test, test_rare_diseases
from src.utils import get_latency, read_data, SAMPLE_DATA_FILE_PATH


class SageLayer(nn.Module):
    def __init__(self, input_size, out_size):
        super(SageLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        
        self.weight = nn.Parameter(torch.FloatTensor(out_size, 2*self.input_size))

        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, features, aggregated_features):
        combined = torch.cat([features, aggregated_features], dim=1)
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined

class GraphSage(nn.Module):
    '''
    Graph Sage Module using Mean aggreation
    '''
    def __init__(self, in_channels, out_channels, embedding_dim, num_layers, num_sample=5):
        super(GraphSage, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_sample = num_sample
        
        self.embedding = nn.Embedding(in_channels, in_channels)
        self.layers = nn.ModuleList()

        for layer in range(self.num_layers - 1):
            self.layers.append(SageLayer(self.in_channels, self.in_channels))
        self.layers.append(SageLayer(self.in_channels, self.out_channels))
        
    def forward(self, nodes, adj_list):
        embedding = None

        for layer in range(self.num_layers):
            if embedding == None:
                embedding = torch.randn(len(nodes), self.in_channels)
                
            aggregated_features = self.aggregate(nodes, adj_list)
            embedding = self.layers[layer](embedding, aggregated_features)
        
        return embedding

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

        column_indices = [i for neighbor in sampled_neighbors for i, n in enumerate(neighbor)]
        row_indices = [i for i in range(len(sampled_neighbors)) for j in range(len(sampled_neighbors[i]))]

        embedding = self.embedding(torch.LongTensor(list(unique_neighbors)))

        mask = torch.zeros(len(sampled_neighbors), len(unique_neighbors))
        mask[row_indices, column_indices] = 1
        num_neighbors = mask.sum(1, keepdim=True)
        num_neighbors[num_neighbors == 0] = 1

        mask = mask.div(num_neighbors)
        aggregate_feats = mask.mm(embedding)
        return aggregate_feats

class GraphSageClassifier(nn.Module):

    def __init__(self, emb_size, num_classes):
        super(GraphSageClassifier, self).__init__()

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
        return self.layer(embeds)


if __name__ == '__main__':
    patients, adj_list, rare_diseases, labels, node_map, train_nodes, test_nodes = read_data(SAMPLE_DATA_FILE_PATH)
    
    in_channels = 10000
    out_channels = 512
    embedding_dim = 1000
    num_layers = 5

    graph_sage = GraphSage(
        in_channels=in_channels,
        out_channels=out_channels,
        embedding_dim=embedding_dim,
        num_layers=num_layers
    )

    classifier = GraphSageClassifier(emb_size=out_channels, num_classes=labels.shape[1])

    start_time = time.time()
    loss_over_time = train(graph_sage, classifier, labels, adj_list, train_nodes, lr=0.01)
    end_time = time.time()

    print(f'Training took {get_latency(start_time, end_time)} seconds.')
    print(f"Cuda memory: ")

    start_time = time.time()
    precision_score, recall_score, f1_score = test(graph_sage, classifier, test_nodes, labels, adj_list)
    end_time = time.time()

    print(f'Testing took {get_latency(start_time, end_time)} seconds.')

    print(f'Stats for all diseases')
    print(f"precision_scores: {precision_score}")
    print(f"recall_score: {recall_score}")
    print(f"f1_score: {f1_score}")

    precision_score, recall_score, f1_score = test_rare_diseases(graph_sage, classifier, test_nodes, labels, adj_list, rare_diseases)
    print(f'Stats for rare diseases')
    print(f"precision_scores: {precision_score}")
    print(f"recall_score: {recall_score}")
    print(f"f1_score: {f1_score}")

    plt.plot(loss_over_time)
    plt.ylabel('GraphSage loss over time')
    plt.show()
    plt.savefig('GraphSage.png')


