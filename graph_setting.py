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





