from typing import Tuple, List, Iterable
from pydot import Dot, graph_from_dot_data, Edge
from graphviz.graphs import BaseGraph
from graphviz import Source
import amrlib
from amrlib.graph_processing.amr_plot import AMRPlot
import numpy as np
import pandas as pd
import csv, pickle


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


