import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt

from src.utils import get_latency, read_data, SAMPLE_DATA_FILE_PATH


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 108, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10))
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(64, 1),
            nn.Softmax())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
    

def train(model, train_nodes, labels, adj_list, node_map, epoch=500, batch_size=50, lr=1e-4):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    model.train()
    loss_over_time = []

    for epoch in range(epoch):
        curr_epoch_loss = []
        batch_nodes = random.sample(train_nodes, batch_size)

        for train_node in batch_nodes:
            inputs = np.zeros(len(node_map))
            inputs[list(adj_list[train_node])] = 1 # mark all neighbors as one

            label = torch.FloatTensor(labels[train_node])
            
            optimizer.zero_grad()
            
            # outputs = model(torch.LongTensor(inputs))
            outputs = model(torch.unsqueeze(torch.FloatTensor(inputs), 0))

            loss = criterion(outputs.squeeze(1), label)
            loss.backward()
            optimizer.step()
            
            curr_epoch_loss.append(loss.cpu().data.numpy())
        print(f"Epoch {epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")
        loss_over_time.append(curr_epoch_loss)
    return loss_over_time

def test(model, test_nodes, labels, adj_list, node_map):
    model.eval()

    true_pos = 0
    false_pos = 0
    false_neg = 0

    for test_node in test_nodes:
        inputs = np.zeros(len(node_map))
        inputs[list(adj_list[test_node])] = 1 # mark all neighbors as one

        disease_indicies = labels[test_node]
        outputs = model(torch.unsqueeze(torch.FloatTensor(inputs), 0))
        outputs = outputs.squeeze(1)

        _, predicted_indicies = torch.topk(outputs.flatten(), disease_indicies.shape[0])
    
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

def test_rare_diseases(model, test_nodes, labels, adj_list, node_map, rare_diseases):
    rare_test_nodes = []
    for test_node in test_nodes:
        if rare_diseases[test_node] == 1:
            rare_test_nodes.append(test_node)
    
    test_nodes = rare_test_nodes

    model.eval()

    true_pos = 0
    false_pos = 0
    false_neg = 0

    for test_node in test_nodes:
        inputs = np.zeros(len(node_map))
        inputs[list(adj_list[test_node])] = 1 # mark all neighbors as one

        disease_indicies = labels[test_node]
        outputs = model(torch.unsqueeze(torch.FloatTensor(inputs), 0))
        outputs = outputs.squeeze(1)

        _, predicted_indicies = torch.topk(outputs.flatten(), disease_indicies.shape[0])
    
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
        

if __name__ == '__main__':
    patients, adj_list, rare_diseases, labels, node_map, train_nodes, test_nodes = read_data(SAMPLE_DATA_FILE_PATH)

    in_features = len(node_map)

    cnn = CNN(in_channels=in_features, out_channels=labels.shape[1])

    start_time = time.time()
    loss_over_time = train(cnn, train_nodes, labels, adj_list, node_map)
    end_time = time.time()

    print(f'Training took {get_latency(start_time, end_time)} seconds.')

    start_time = time.time()
    precision_score, recall_score, f1_score = test(cnn, test_nodes, labels, adj_list, node_map)
    end_time = time.time()

    print(f'Testing took {get_latency(start_time, end_time)} seconds.')

    print(f'Stats for all diseases')
    print(f"precision_scores: {precision_score}")
    print(f"recall_score: {recall_score}")
    print(f"f1_score: {f1_score}")

    precision_score, recall_score, f1_score = test_rare_diseases(cnn, test_nodes, labels, adj_list, node_map, rare_diseases)
    print(f'Stats for rare diseases')
    print(f"precision_scores: {precision_score}")
    print(f"recall_score: {recall_score}")
    print(f"f1_score: {f1_score}")

    plt.plot(loss_over_time)
    plt.ylabel('CNN loss over time')
    plt.show()
    plt.savefig('CNN.png')


