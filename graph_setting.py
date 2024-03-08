from typing import Tuple, List, Iterable
from pydot import Dot, graph_from_dot_data, Edge
from graphviz.graphs import BaseGraph
from graphviz import Source

import amrlib
from amrlib.graph_processing.amr_plot import AMRPlot

import numpy as np
import pandas as pd
import csv, pickle

import torch
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def get_graph_dot_obj(graph_spec) -> List[Dot]:
    _original_graph_spec = graph_spec
    if isinstance(graph_spec, (BaseGraph, Source)):
        graph_spec = graph_spec.source
    if isinstance(graph_spec, str):
        graph_spec = graph_from_dot_data(graph_spec)

    assert isinstance(graph_spec, list) and all(
        isinstance(x, Dot) for x in graph_spec
    ), (
        f"Couldn't get a proper dot object list from: {_original_graph_spec}. "
        f"At this point, we should have a list of Dot objects, but was: {graph_spec}"
    )
    return graph_spec


def get_edges(graph_spec, label=False):
    graphs = get_graph_dot_obj(graph_spec)
    n_graphs = len(graphs)
    if n_graphs > 1:
        return [get_edges(graph) for graph in graphs]
    elif n_graphs == 0:
        raise ValueError(f"Your input had no graphs")
    else:
        graph = graphs[0]
        edges = graph.get_edges()
        edges_list = []
        if not label:
            for edge in edges:
                r1, r2 = graph.get_node(edge.get_source())[0].get_label().strip('\"').strip('\\').strip('\"'), \
                         graph.get_node(edge.get_destination())[0].get_label().strip('\"').strip('\\').strip('\"')
                if '/' in r1:
                    r1 = r1.split('/')[1]
                elif '\\' in r1:
                    r1 = r1.split('\\')[0]

                if '/' in r2:
                    r2 = r2.split('/')[1]
                elif '\\' in r1:
                    r2 = r2.split('\\')[0]

                edges_list.append([r1, r2])
        else:
            for edge in edges:
                r1, r2, r3 = graph.get_node(edge.get_source())[0].get_label().strip('\"').strip('\\').strip('\"'), \
                             graph.get_node(edge.get_destination())[0].get_label().strip('\"').strip('\\').strip(
                                 '\"'), edge.get_label().strip('\"')[1:]
                if '/' in r1:
                    r1 = r1.split('/')[1]
                elif '\\' in r1:
                    r1 = r1.split('\\')[0]

                if '/' in r2:
                    r2 = r2.split('/')[1]
                elif '\\' in r1:
                    print("called")
                    r2 = r2.split('\\')[0]

                edges_list.append([r1, r2, r3])

        return edges_list

gtrs = np.load('g_train.npy',allow_pickle=True)
gtes = np.load('g_test.npy',allow_pickle=True)
gall = np.concatenate((gtrs, gtes), axis=0)
print(gtrs.shape, gtes.shape, gall.shape, type(gtrs))
print(gtrs[4])

word_set = list({ts[i] for g in gall for ts in g for i in range(2)})
edge_set = list({ts[2] for g in gall for ts in g})
word_set.sort()
edge_set.sort()
word_to_id = dict(zip(word_set,[i for i in range(len(word_set))]))
edge_to_id = dict(zip(edge_set,[i for i in range(len(edge_set))]))
Vsize, Esize = len(word_to_id), len(edge_to_id)


def data_embedding(edges):
    nodes = list({edge[i] for edge in edges for i in range(2)})
    nodes_to_id = dict(zip(nodes,[i for i in range(len(nodes))]))
    edge_index = [[nodes_to_id[edge[0]] for edge in edges], [nodes_to_id[edge[1]] for edge in edges]]
    x, edge_attr = [], []
    for node in nodes_to_id.keys():
        vector = np.zeros(Vsize)
        vector[word_to_id[node]] = 1.0
        x.append(vector)

    for edge in edges:
        vector = np.zeros(Esize)
        vector[edge_to_id[edge[2]]] = 1.0
        edge_attr.append(vector)

    return np.array(x), np.array(edge_index), np.array(edge_attr)


def label_converter(labels):
    nlabels = []
    for label in labels:
        if label == 'Positive':
            nlabels.append(2)
        elif label == 'Negative':
            nlabels.append(0)
        else:
            nlabels.append(1)
    return nlabels


def get_dataset(graph,labels):
    dataset = []
    for i in range(len(graph)):
        x, edge_index, edge_attr = data_embedding(graph[i])
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=[labels[i]]))
    return dataset


train_dataset = get_dataset(gtrs,y_train)
test_dataset = get_dataset(gtes, y_test)
print(len(train_dataset), len(test_dataset))
print(train_dataset[4], test_dataset[48])


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(Vsize, 8)
        self.conv2 = GCNConv(8, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()

for epoch in tqdm(range(50), total=50):
    avg_loss = []
    for i in range(1600):
        if not gtrs[i]: # ignore empty graph
            continue
        data = train_dataset[i]
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, torch.tensor(data.y*np.ones(out.shape[0]), dtype=torch.long))
        loss.backward()
        avg_loss.append(loss.item())
        optimizer.step()
    print(np.average(avg_loss))
