"""Structure-type assignment for SWCs."""

from __future__ import annotations

import networkx as nx
from networkx.algorithms.dag import dag_longest_path

from aind_morphology_utils.swc import NeuronGraph, StructureTypes


def fix_structure_assignment(swc_in_path: str, swc_out_path: str) -> None:
    """Assign axon vs basal dendrite from longest soma child branch."""
    graph = NeuronGraph.from_swc(swc_in_path)

    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    if not roots:
        raise ValueError(f"No root/soma found for {swc_in_path}")

    soma = roots[0]
    graph.nodes[soma]["struct_type"] = StructureTypes.SOMA.value

    soma_children = list(graph.successors(soma))
    if not soma_children:
        graph.save_swc(swc_out_path)
        return

    def mark_subtree_iterative(start, structure_type: int) -> None:
        stack = [start]
        while stack:
            node = stack.pop()
            graph.nodes[node]["struct_type"] = structure_type
            stack.extend(list(graph.successors(node)))

    axon_branch = None
    longest_branch_length = -1

    for child in soma_children:
        branch_nodes = set(nx.descendants(graph, child))
        branch_nodes.add(child)
        path = dag_longest_path(graph.subgraph(branch_nodes))
        path_length = len(path)
        if path_length > longest_branch_length:
            longest_branch_length = path_length
            axon_branch = child

    if axon_branch is None:
        raise ValueError(f"Could not determine axon branch for {swc_in_path}")

    for child in soma_children:
        if child == axon_branch:
            mark_subtree_iterative(child, StructureTypes.AXON.value)
        else:
            mark_subtree_iterative(child, StructureTypes.BASAL_DENDRITE.value)

    graph.save_swc(swc_out_path)
