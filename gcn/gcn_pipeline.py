import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
from google.colab import drive
import networkx as nx
import json
import time
import random
import copy
import tiktoken
import random
import os

os.environ["DGLBACKEND"] = "pytorch"

import torch
import dgl
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch import GraphConv
from sklearn.model_selection import train_test_split

from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info


def list_duplicates(data):
    seen = set()
    duplicates = []

    duplicates.append(set( x['id'] for x in data if x['id'] in seen or seen.add(x['id']) ))

    return list( duplicates )


def process_node_properties_with_missing_values(node_properties):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Define the fixed length for embeddings
    FIXED_LENGTH = 1024  # Example fixed length
    concatenated_properties = ""
    for key, value in node_properties.items():
        if value is None:
            value = "missing"
        concatenated_properties += f"{key}:{value} "

    # Encode the concatenated string
    embedding = model.encode(concatenated_properties)
    embedding = list(embedding)
    # Ensure the embedding is of fixed length
    if len(embedding) > FIXED_LENGTH:
        embedding = embedding[:FIXED_LENGTH]
    else:
        embedding = embedding + [0] * (FIXED_LENGTH - len(embedding))

    return torch.tensor(embedding, dtype=torch.float)


def prep_for_gcn(sample):
    # Create an empty NetworkX graph
    nx_graph = nx.Graph()

    # Add nodes to the graph
    for node in sample:
        nx_graph.add_node(node['id'], **node['properties'])

    # Verify node IDs in relationships before adding edges
    for node in sample:
        for relationship in node['relationships']:
            if 'target_id' in relationship.keys():
                if nx_graph.has_node(relationship['target_id']):
                    nx_graph.add_edge(node['id'], relationship['target_id'], type=relationship['type'])
                else:
                    print(f"Warning: target_id {relationship['target_id']} for node_id {node['id']} does not exist in the graph")

    print(f"Graph has {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges")

    # Create DGL graph for gnn
    dgl_graph = dgl.from_networkx(nx_graph)

    # Extract node properties to add as features
    node_features = []
    labels = []
    node_ids = set(nx_graph.nodes())

    for node in sample:
        if node['id'] in node_ids:
            node_data = node['properties']
            node_feature = process_node_properties_with_missing_values(node_data)
            node_features.append(node_feature)

            label = node['label']
            if label == "Valid":
                labels.append(0)
            elif label == "Invalid":
                labels.append(1)

    # Ensure the number of node features matches the number of nodes
    if len(node_features) != len(node_ids):
        raise ValueError(f"Number of features ({len(node_features)}) does not match number of nodes ({len(node_ids)})")

    # Stack tensors to create a feature matrix
    node_features = torch.stack(node_features)
    labels = torch.tensor(labels, dtype=torch.long)

    # Add node features & labels to the DGL graph
    dgl_graph.ndata['feat'] = node_features
    dgl_graph.ndata['label'] = labels

    return dgl_graph



class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.dropout = nn.Dropout(p=0.4)  # Add dropout layer

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.dropout(h)  # Apply dropout
        h = self.conv2(g, h)
        return h

# Set random seeds for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)



def main():
    dataset = labeled_data_amazon_fine_foods_12_shot + bad_data + bad_relationships

    for c in dataset:
        if 'label' in c.keys():
            continue
    else:
        c['label'] = 'Invalid'

    # Manually set the training and testing masks based on known node IDs
    train_node_ids_12_shot = ['F001E4KFGX', 'E001E4KFGX', 'D001E4KFGX', 'A001E4KFG1', 'C001E4KFG1', 'B001E4KFG1', "B001E4KFG0_F2SGXH7AUHU8GW_1303862400", "B001E4KFG0_E2SGXH7AUHU8GW_1303862400",
                          "B001E4KFG0_D2SGXH7AUHU8GW_1303862400", "B001E4KFG0_C5GH6H7AUHU8GW_1303862400", "B001E4KFG0_B5GH6H7AUHU8GW_1303862400", "B001E4KFG0_A5GH6H7AUHU8GW_1303862400"]  # Known training node IDs
    train_node_ids_4_shot = ["B001E4KFG1", "B001E4KFG0_A2SGXH7AUHU8GW_1303862400", "B001E4KFGX", "B001E4KFG0_A5GH6H7AUHU8GW_1303862400"]  # Known training node IDs
    train_mask = torch.zeros(len(dataset), dtype=torch.bool)
    test_mask = torch.zeros(len(dataset), dtype=torch.bool)
    val_mask = torch.zeros(len(dataset), dtype=torch.bool)

    # Create a mapping from node ID to index
    id_to_idx = {node['id']: idx for idx, node in enumerate(dataset)}

    # Set the training mask
    for node_id in train_node_ids_12_shot:
        if node_id in id_to_idx:
            train_mask[id_to_idx[node_id]] = True
            counter = 0

    # Set the val mask (all other nodes)
    for idx in range(len(dataset)):
        if (not train_mask[idx]) and counter < 22:
            val_mask[idx] = True
            counter += 1

    # Set the testing mask (all other nodes)
    for idx in range(len(dataset)):
        if (not train_mask[idx]) and (not val_mask[idx]):
            test_mask[idx] = True

    # Verify the mask lengths
    print(f"Train Mask Length: {train_mask.sum().item()}")
    print(f"Val Mask Length: {val_mask.sum().item()}")

    combined_dgl_graph = prep_for_gcn(dataset) # Your graph initialization here
    combined_dgl_graph = dgl.add_self_loop(combined_dgl_graph)

    # Example seed value
    seed = 42
    set_random_seed(seed)

    # Initialize the GCN model
    in_feats = combined_dgl_graph.ndata['feat'].shape[1]
    h_feats = 6 # Adjust based on your model complexity needs
    num_classes = len(torch.unique(combined_dgl_graph.ndata['label']))

    model = GCN(in_feats, h_feats, num_classes)

    # Set up optimizer and loss function
    learning_rate = 0.01  # Adjust based on performance
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Training loop with early stopping
    num_epochs = 20  # Start with a small number of epochs
    best_val_acc = 0
    patience = 4  # Number of epochs to wait before early stopping
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        logits = model(combined_dgl_graph, combined_dgl_graph.ndata['feat'])
        loss = loss_fn(logits[train_mask], combined_dgl_graph.ndata['label'][train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_logits = model(combined_dgl_graph, combined_dgl_graph.ndata['feat'])
            val_pred = val_logits[val_mask].argmax(dim=1)
            val_loss = loss_fn(val_logits[val_mask], combined_dgl_graph.ndata['label'][val_mask])
            val_labels = combined_dgl_graph.ndata['label'][val_mask]
            val_correct = (val_pred == val_labels).sum().item()
            val_acc = val_correct / val_mask.sum().item()

        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)

        # Check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Accuracy: {val_acc:.4f}')

    # Load the best model
    #model.load_state_dict(best_model)

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        logits = model(combined_dgl_graph, combined_dgl_graph.ndata['feat'])
        pred = logits.argmax(dim=1)
        test_pred = pred[test_mask]
        test_labels = combined_dgl_graph.ndata['label'][test_mask]
        correct = (test_pred == test_labels).sum().item()
        acc = correct / test_mask.sum().item()
        print(f'Test Accuracy: {acc:.4f}')

    # Make predictions on new data
    model.eval()
    with torch.no_grad():
        logits = model(combined_dgl_graph, combined_dgl_graph.ndata['feat'])
        pred = logits.argmax(dim=1)

    # Get the predicted labels
    predicted_labels = pred.cpu().numpy()
    print(predicted_labels) 

if __name__ == "__main__":
    main()  