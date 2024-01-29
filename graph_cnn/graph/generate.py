import random
from statistics import mean
import igraph as ig
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from networkx.algorithms.traversal.depth_first_search import dfs_tree


def create_random_graph(nodes: int, edges: int) -> nx.DiGraph:
    """
    Creates a random directed graph with the specified number of nodes and edges.

    Args:
        nodes (int): The number of nodes in the graph.
        edges (int): The number of edges in the graph.

    Returns:
        nx.DiGraph: A random directed graph with the specified number of nodes and edges.
    """
    graph = ig.Graph.Erdos_Renyi(n=nodes, p=edges, directed=False, loops=False)
    graph.to_directed(mode="acyclic")
    graph = graph.to_networkx()
    return graph


def swap_sub_trees(t1, node1, t2, node2):
    """
    Swaps the sub-trees rooted at `node1` in tree `t1` with the sub-trees rooted at `node2` in tree `t2`.

    Args:
        t1 (Tree): The first tree.
        node1 (Node): The root node of the sub-tree in `t1` to be swapped.
        t2 (Tree): The second tree.
        node2 (Node): The root node of the sub-tree in `t2` to be swapped.

    Returns:
        None
    """
    sub_t1 = dfs_tree(t1, node1).edges()
    sub_t2 = dfs_tree(t2, node2).edges()
    replace_sub_tree(t1, sub_t1, node1, sub_t2, node2)
    replace_sub_tree(t2, sub_t2, node2, sub_t1, node1)


def replace_sub_tree(t, sub_t_remove, root_remove, sub_t_add, root_add):
    """
    Replaces a subtree in a graph with another subtree.

    Parameters:
    t (networkx.Graph): The graph in which the subtree is to be replaced.
    sub_t_remove (list): List of edges to be removed from the original subtree.
    root_remove (int): The root node of the original subtree to be removed.
    sub_t_add (list): List of edges to be added for the new subtree.
    root_add (int): The root node of the new subtree to be added.
    """
    t.remove_edges_from(sub_t_remove)
    t.add_edges_from(sub_t_add)
    in_edges = list(t.in_edges(nbunch=root_remove))
    t.remove_edges_from(in_edges)
    t.add_edges_from([(edge[0], root_add) for edge in in_edges])


def cross_over(graph1, graph2):
    """
    Perform crossover operation between two graphs.

    Args:
        graph1 (networkx.Graph): The first graph.
        graph2 (networkx.Graph): The second graph.

    Returns:
        tuple: A tuple containing the modified graph1 and graph2.
    """
    node1 = np.random.choice(list(graph1.nodes()))
    node2 = np.random.choice(list(graph2.nodes()))
    swap_sub_trees(graph1, node1, graph2, node2)
    return graph1, graph2


def mutate(graph):
    """
    Mutates the given graph by randomly adding or removing nodes or edges.

    Parameters:
    graph (networkx.Graph): The graph to be mutated.

    Returns:
    networkx.Graph: The mutated graph.
    """
    choice = np.random.choice(["add_node", "remove_node", "add_edge", "remove_edge"])
    if choice == "add_node":
        node_type = np.random.choice(
            [
                "Convolution",
                "MaxPooling",
                "AveragePooling",
                "LocallyConnected2D",
                "Activation",
            ]
        )
        graph.add_node(len(graph.nodes()), state=node_type)
        if node_type == "Convolution":
            kernel_size = (np.random.choice([1, 3, 5, 7]),) * 2
            activation = np.random.choice(["elu", "leaky_relu", "tanh"])
            activation = "relu"
            graph.nodes[len(graph.nodes()) - 1]["kernel_size"] = kernel_size
            graph.nodes[len(graph.nodes()) - 1]["activation"] = activation
        elif node_type == "MaxPooling" or node_type == "AveragePooling":
            kernel_size = (np.random.choice([2, 3, 4]),) * 2
            activation = np.random.choice(["elu", "leaky_relu", "tanh"])
            activation = "relu"
            graph.nodes[len(graph.nodes()) - 1]["kernel_size"] = kernel_size
            graph.nodes[len(graph.nodes()) - 1]["activation"] = activation
    elif choice == "remove_node":
        graph.remove_node(np.random.choice(list(graph.nodes())))
    elif choice == "add_edge":
        nodes = random.choices(list(graph.nodes()), k=2)
        nodes.sort()
        graph.add_edge(nodes[0], nodes[1])
    elif choice == "remove_edge":
        nodes = random.choices(list(graph.edges()), k=1)[0]
        graph.remove_edge(nodes[0], nodes[1])
    return graph


def assign_states(graph):
    """
    Assigns random layer types, activations, and kernel/pool sizes to each node in the graph.

    Parameters:
    graph (networkx.Graph): The graph to assign states to.

    Returns:
    networkx.Graph: The graph with assigned states.
    """
    for node in graph.nodes():
        layer_type = np.random.choice(
            ["Convolution", "LocallyConnected2D", "MaxPooling"]
        )
        graph.nodes[node]["layer_type"] = layer_type

        if layer_type == "Convolution":
            activation = np.random.choice(["relu", "sigmoid", "tanh"])
            graph.nodes[node]["activation"] = activation
            kernel_size = (np.random.choice([1, 3, 5]),) * 2
            graph.nodes[node]["kernel_size"] = kernel_size
            

        elif layer_type == "LocallyConnected2D":
            activation = np.random.choice(["relu", "sigmoid", "tanh"])
            graph.nodes[node]["activation"] = activation
            kernel_size = (np.random.choice([1, 3, 5]),) * 2
            graph.nodes[node]["kernel_size"] = kernel_size

        elif layer_type == "MaxPooling":
            activation = np.random.choice(["relu", "sigmoid", "tanh"])
            pool_size = (np.random.choice([2, 3, 4]),) * 2
            graph.nodes[node]["kernel_size"] = pool_size
            graph.nodes[node]["activation"] = activation

    return graph


def final_graph(graph):
    """
    Remove isolated nodes from the graph and assign states to the remaining nodes.

    Parameters:
    graph (networkx.Graph): The input graph.

    Returns:
    networkx.Graph: The modified graph with isolated nodes removed and states assigned.
    """
    isolated = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated)
    graph = assign_states(graph)
    return graph


def create_final_graph(nodes: int, edges: int) -> nx.DiGraph:
    """
    Creates a final graph with the specified number of nodes and edges.

    Args:
        nodes (int): The number of nodes in the graph.
        edges (int): The number of edges in the graph.

    Returns:
        nx.DiGraph: The final graph with the specified number of nodes and edges.
    """
    graph = create_random_graph(nodes, edges)
    graph = final_graph(graph)
    return graph


if __name__ == "__main__":
    graph = create_final_graph(10, 0.1)
    nx.draw(graph, with_labels=True)
    plt.show()
