import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train(graph_model, classifier, labels, adj_list, train_nodes, batch_size=50, epochs=8000, lr=0.01):
    print('Training classifier ...')
    models = [graph_model, classifier]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params, lr=lr)

    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    for epoch in range(epochs):

        batch_nodes = random.sample(train_nodes, batch_size)

        predicted_embedding = graph_model(batch_nodes, adj_list)
        predicted_labels = classifier(predicted_embedding)

        expected_labels = torch.FloatTensor(labels[np.array(batch_nodes, dtype=np.int64)])
        loss = criterion(predicted_labels, expected_labels)
        
        optimizer.zero_grad()
        for model in models:
            model.zero_grad()

        loss.backward()
   
        optimizer.step()

        print(epoch, loss.data)
