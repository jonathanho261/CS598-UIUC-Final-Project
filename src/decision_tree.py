import numpy as np
from sklearn import tree

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

from src.utils import get_latency, read_data, SAMPLE_DATA_FILE_PATH

'''
Used to represent Decision Tree

'''

def setup_data(nodes, adj_list, node_map, labels):
    '''
    Represent diseases as samples -> node has a disease (first one seen), it will be of that class
    Represent neighbors as features 0 = not neighbor, 1 = neighbor
    '''
    n_samples = len(nodes)
    n_features = len(node_map)

    X = np.zeros((n_samples, n_features))
    Y = np.zeros(n_samples)
    for i, node in enumerate(nodes):
        X[i, list(adj_list[node])] = 1
        Y[i] = np.argmax(labels[nodes])
    return X, Y


if __name__ == '__main__':
    patients, adj_list, rare_diseases, labels, node_map, train_nodes, test_nodes = read_data(SAMPLE_DATA_FILE_PATH)
    
    print("Training Decision Tree")
    start_time = time.time()
    X_train, Y_train = setup_data(train_nodes, adj_list, node_map, labels)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)
    end_time = time.time()

    print(f'Testing took {get_latency(start_time, end_time)} seconds.')

    print("Testing Decision Tree")
    start_time = time.time()
    X_test, Y_test = setup_data(test_nodes, adj_list, node_map, labels)
    Y_hat = clf.predict(X_test)
    end_time = time.time()

    print(f'Testing took {get_latency(start_time, end_time)} seconds.')

    print(f'Stats for decision tree')
    print(f"accuracy_score: {accuracy_score(Y_test, Y_hat)}")
    print(f"precision_scores: {precision_score(Y_test, Y_hat, pos_label=12.0)}")
    print(f"recall_score: {recall_score(Y_test, Y_hat, pos_label=12.0)}")
    print(f"f1_score: {f1_score(Y_test, Y_hat, pos_label=12.0)}")










