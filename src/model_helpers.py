import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def train(graph_model, classifier, labels, adj_list, train_nodes, batch_size=50, epochs=2000, lr=0.01):
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

    
    loss_over_time = []
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
        loss_over_time.append(loss.data.item())
    return loss_over_time

def test(graph_model, classifier, test_nodes, labels, adj_list):
    print('Testing classifier ...')
    target = torch.LongTensor(labels[test_nodes])

    embedding = graph_model(test_nodes, adj_list)
    output = classifier(embedding)

    true_pos = 0
    false_pos = 0
    false_neg = 0

    for i in range(target.shape[0]):
        disease_indicies = torch.nonzero(target[i]).flatten()
        _, predicted_indicies = torch.topk(output[i], disease_indicies.shape[0])

        for disease in disease_indicies:
            if disease in predicted_indicies:
                true_pos += 1
            else:
                false_neg += 1
        
        for prediction in predicted_indicies:
            if prediction not in disease_indicies:
                false_pos += 1

    precision_score = (true_pos)/(true_pos+false_pos) if true_pos+false_pos > 0 else 0
    recall_score = (true_pos)/(true_pos+false_neg) if true_pos+false_neg > 0 else 0
    f1_score = (2*precision_score*recall_score)/(precision_score+recall_score) if precision_score+recall_score > 0 else 0

    return precision_score, recall_score, f1_score

def test_rare_diseases(graph_model, classifier, test_nodes, labels, adj_list, rare_diseases):
    print('Testing rare_diseases classifier ...')
    rare_test_nodes = []
    for test_node in test_nodes:
        if rare_diseases[test_node] == 1:
            rare_test_nodes.append(test_node)
    
    test_nodes = rare_test_nodes

    target = torch.LongTensor(labels[test_nodes])

    embedding = graph_model(test_nodes, adj_list)
    output = classifier(embedding)

    true_pos = 0
    false_pos = 0
    false_neg = 0

    for i in range(target.shape[0]):
        disease_indicies = torch.nonzero(target[i]).flatten()
        _, predicted_indicies = torch.topk(output[i], disease_indicies.shape[0])

        for disease in disease_indicies:
            if disease in predicted_indicies:
                true_pos += 1
            else:
                false_neg += 1
        
        for prediction in predicted_indicies:
            if prediction not in disease_indicies:
                false_pos += 1

    precision_score = (true_pos)/(true_pos+false_pos) if true_pos+false_pos > 0 else 0
    recall_score = (true_pos)/(true_pos+false_neg) if true_pos+false_neg > 0 else 0
    f1_score = (2*precision_score*recall_score)/(precision_score+recall_score) if precision_score+recall_score > 0 else 0

    return precision_score, recall_score, f1_score
