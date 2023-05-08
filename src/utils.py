import pickle

import os
print(os.getcwd())

SAMPLE_DATA_FILE_PATH = './data/sample_graph'

def read_data(file_path):
    '''
    D = number of diseases
    S = number of symptoms

    @return patients: list
    @return adj_list: {int: set(neighbors)}
    @return is_rare_disease: NumPy array of shape (D * 1)
    @return symptoms: NumPy array of shape (D * S)
    '''
    patients = pickle.load(open(file_path + ".nodes.pkl", "rb"))
    adj_list = pickle.load(open(file_path + ".adj.pkl", "rb"))
    rare_diseases = pickle.load(open(file_path + ".rare.label.pkl", "rb"))
    labels = pickle.load(open(file_path + ".label.pkl", "rb"))
    node_map = pickle.load(open(file_path + ".map.pkl", "rb"))
    train_nodes = pickle.load(open(file_path + ".train.pkl", "rb"))
    test_nodes = pickle.load(open(file_path + ".test.pkl", "rb"))
    
    return patients, adj_list, rare_diseases, labels, node_map, train_nodes, test_nodes
