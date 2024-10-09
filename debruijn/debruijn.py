#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
from pathlib import Path
import networkx as nx
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
)
import matplotlib
from operator import itemgetter
import random

random.seed(9001)
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
from typing import Iterator, Dict, List, Tuple

matplotlib.use("Agg")

__author__ = "HaidingWANG"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["HaidingWANG"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "HaidingWANG"
__email__ = "wanghaidingfr@126.com"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, 'r') as file:
        while True:
            file.readline()
            sequence = file.readline().strip()
            file.readline()
            file.readline()
            if len(sequence) == 0:
                break
            yield sequence


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i:i+kmer_size]


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = {}
    for read in read_fastq(fastq_file):
        for kmer in cut_kmer(read, kmer_size):
            if kmer in kmer_dict:
                kmer_dict[kmer] += 1
            else:
                kmer_dict[kmer] = 1
    return kmer_dict
    


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph  = DiGraph()

    for kmer, count in kmer_dict.items():
        prefix = kmer[:-1]
        suffix = kmer[1:]

        if graph.has_edge(prefix, suffix):
            graph[prefix][suffix]['weight'] += count
        else:
            graph.add_edge(prefix, suffix, weight = count)
    
    return graph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        if delete_entry_node and delete_sink_node:
            graph.remove_nodes_from(path)
        elif delete_entry_node:
            graph.remove_nodes_from(path[:-1])
        elif delete_sink_node:
            graph.remove_nodes_from(path[1:])
        else:
            graph.remove_nodes_from(path[1:-1]) 
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    weight_stdev = statistics.stdev(weight_avg_list) if len(weight_avg_list) > 1 else 0

    if weight_stdev > 0:
        best_index = weight_avg_list.index(max(weight_avg_list))
    else:
        length_stdev = statistics.stdev(path_length) if len(path_length) > 1 else 0
        if length_stdev > 0:
            best_index = path_length.index(max(path_length))
        else:
            best_index = random.randint(0, len(path_list) - 1)
    
    best_path = path_list[best_index]
    paths_to_remove = [path for i, path in enumerate(path_list) if i != best_index]
    remove_paths(graph, paths_to_remove, delete_entry_node, delete_sink_node)
    
    return graph


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    paths = list(all_simple_paths(graph, ancestor_node, descendant_node))
    path_length = [len(path) for path in paths]
    weight_avg_list = [path_average_weight(graph, path) for path in paths]

    graph = select_best_path(graph, paths, path_length, weight_avg_list, delete_entry_node=False, delete_sink_node=False)

    return graph


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble = False
    count = 0
    for node in graph.nodes:
        list_predecesseurs = list(graph.predecessors(node))
        if len(list_predecesseurs) > 1:
            for i in range(len(list_predecesseurs)):
                for j in range(i+1, len(list_predecesseurs)):
                    ancestor_node = nx.lowest_common_ancestor(graph, list_predecesseurs[i], list_predecesseurs[j])
                    if ancestor_node is not None:
                        bubble = True
                        break
            if bubble:
                break

    if bubble:
        count +=1
        graph = simplify_bubbles(solve_bubble(graph, ancestor_node, node))
        return graph

    return graph


def solve_tip_to_entry(graph: DiGraph, ancestors: List[str], upnode: str, delete_entry_node: bool = False, delete_sink_node: bool = False) -> DiGraph:
    """
    Remove unwanted paths (entrey_tips) from the graph.

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestors: A list of upstream ancestor nodes 
    :param upnode: The upstram node 
    :return: The modified graph
    """
    path_list = []
    for ancestor in ancestors:
        path_list.extend(all_simple_paths(graph, ancestor, upnode))
    path_length = [len(path) for path in path_list]
    weight_average_list = []
    for path in path_list:
        weight_average_list.append(path_average_weight(graph, path))
    return select_best_path(graph, path_list, path_length, weight_average_list,delete_entry_node, delete_sink_node)

def solve_tip_to_sink(graph: DiGraph, successtors: List[str], downnode: str, delete_entry_node: bool = False, delete_sink_node: bool = False) -> DiGraph:
    """
    Remove unwanted paths (out_tips) from the graph.

    :param graph: (nx.DiGraph) A directed graph object
    :param successtors: A list of upstream ancestor nodes 
    :param downnode: The upstram node 
    :return: The modified graph
    """
    path_list = []
    for successtor in successtors:
        path_list.extend(all_simple_paths(graph, downnode, successtor))
    path_length = [len(path) for path in path_list]
    weight_average_list = []
    for path in path_list:
        weight_average_list.append(path_average_weight(graph, path))
    return select_best_path(graph, path_list, path_length, weight_average_list,delete_entry_node, delete_sink_node)


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    node_n = None
    ancestors = []

    for node in graph.nodes:
        pred = list(graph.predecessors(node))

        if len(pred) > 1:
            for start in starting_nodes:
                if (start in graph) and (nx.has_path(graph, start, node)):
                    ancestors.append(start)
            if len(ancestors) >= 2:
                node_n = node
                break
    if len(ancestors) >= 2:
        graph = solve_tip_to_entry(graph, ancestors, node_n, True, False)
        graph = solve_entry_tips(graph, starting_nodes)

    return graph


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    node_n = None
    successors = []

    for node in graph.nodes:
        succ = list(graph.successors(node))

        if len(succ) > 1:
            for end in ending_nodes:
                if (end in graph) and (has_path(graph, node, end)):
                    successors.append(end)
            if (len(successors) >= 2):
                node_n = node
                break
    if len(successors) >= 2:
        graph = solve_tip_to_sink(graph, successors, node_n, False, True)
        graph = solve_out_tips(graph, ending_nodes)

    return graph



def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    return [node for node in graph.nodes if len(list(graph.predecessors(node))) == 0]


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    return [node for node in graph.nodes if len(list(graph.successors(node))) == 0]


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs_list = []
    for starting_node in starting_nodes:
        for ending_node in ending_nodes:
            if starting_node in graph and ending_node in graph:
                if has_path(graph, starting_node, ending_node):
                    paths = all_simple_paths(graph, starting_node, ending_node)
                    for path in paths:
                        contig = path[0]
                        for node in path[1:]:
                            contig += node[-1]
                        contigs_list.append((contig, len(contig)))
                        print(f"Contig generated: {contig} of length {len(contig)}")
            else:
                print(f"Warning: Node {starting_node} or {ending_node} not in graph.")
    return contigs_list



def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, 'w') as file:
        for i, (contig, length) in enumerate(contigs_list):
            file.write(f">contig_{i} len={length}\n")
            file.write(textwrap.fill(contig, width = 80)+ "\n")
    


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Read the Fastq file and build the k-mer dictionary
    print("Building k-mer dictionary...")
    kmer_dict = build_kmer_dict(args.fastq_file, args.kmer_size)

    # Build the De Bruijn graph
    print("Building De Bruijn graph...")
    graph = build_graph(kmer_dict)

    # Solve bubbles in the graph
    print("Solving bubbles...")
    graph = simplify_bubbles(graph)

    # Identify and solve entry and out tips
    print("Solving entry tips...")
    starting_nodes = get_starting_nodes(graph)
    graph = solve_entry_tips(graph, starting_nodes)

    print("Solving out tips...")
    ending_nodes = get_sink_nodes(graph)
    graph = solve_out_tips(graph, ending_nodes)

    # Extract contigs
    print("Extracting contigs...")
    contigs_list = get_contigs(graph, starting_nodes, ending_nodes)

    # Write the contigs to the output file
    print(f"Writing contigs to {args.output_file}...")
    save_contigs(contigs_list, args.output_file)

    # Optionally, draw the graph if graph image file is provided
    if args.graphimg_file:
        print(f"Drawing graph and saving as {args.graphimg_file}...")
        draw_graph(graph, args.graphimg_file)

    print("Program finished.")


if __name__ == "__main__":  # pragma: no cover
    main()

