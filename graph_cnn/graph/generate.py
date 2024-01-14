import igraph as ig
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import random
from networkx.algorithms.traversal.depth_first_search import dfs_tree


def create_random_graph(nodes:int,edges:int)->nx.DiGraph:
    graph = ig.Graph.Erdos_Renyi(n=nodes, p=edges, directed=False, loops=False)
    graph.to_directed(mode='acyclic')
    graph = graph.to_networkx()
    return graph

def swap_sub_trees(t1, node1, t2, node2):
    sub_t1 = dfs_tree(t1, node1).edges()
    sub_t2 = dfs_tree(t2, node2).edges()
    replace_sub_tree(t1, sub_t1, node1, sub_t2, node2)
    replace_sub_tree(t2, sub_t2, node2, sub_t1, node1)

def replace_sub_tree(t, sub_t_remove, root_remove, sub_t_add, root_add):
    t.remove_edges_from(sub_t_remove)
    t.add_edges_from(sub_t_add)
    in_edges = list(t.in_edges(nbunch=root_remove))
    t.remove_edges_from(in_edges)
    t.add_edges_from([(edge[0], root_add) for edge in in_edges])
    
def cross_over(graph1,graph2):
    node1 = np.random.choice(list(graph1.nodes()))
    node2 = np.random.choice(list(graph2.nodes()))
    swap_sub_trees(graph1,node1,graph2,node2)
    return graph1,graph2

def mutate(graph):
    choice = np.random.choice(["add_node","remove_node","add_edge","remove_edge"])
    if choice == "add_node":
        node_type = np.random.choice(["Convolution","MaxPooling","AveragePooling"])
        graph.add_node(len(graph.nodes()), state=node_type)
        if node_type == 'Convolution':
            kernel_size = (np.random.choice([1, 3, 5, 7]),) * 2
            activation = np.random.choice(["elu", "leaky_relu", "tanh"])
            graph.nodes[len(graph.nodes()) - 1]['kernel_size'] = kernel_size
            graph.nodes[len(graph.nodes()) - 1]['activation'] = activation
        elif node_type == "MaxPooling" or node_type == "AveragePooling":
            kernel_size = (np.random.choice([2, 3, 4]),) * 2
            activation = np.random.choice(["elu", "leaky_relu", "tanh"])
            graph.nodes[len(graph.nodes()) - 1]['kernel_size'] = kernel_size
            graph.nodes[len(graph.nodes()) - 1]['activation'] = activation
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
    for node in graph.nodes():
        state = np.random.choice(["MaxPooling","AveragePooling","Convolution"],p=[0.05,0.05,0.9])
        activation = np.random.choice(["relu","sigmoid","tanh"])
        graph.nodes[node]['state'] = state
        if state == "Convolution":
            graph.nodes[node]['kernel_size'] = (np.random.choice([1,3,5]),)*2
            graph.nodes[node]['activation'] = activation
        elif state == "MaxPooling" or state == "AveragePooling":
            graph.nodes[node]['kernel_size'] = (np.random.choice([2,3,4]),)*2
            graph.nodes[node]['activation'] = activation
            
    return graph

def final_graph(graph):
    isolated = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated)
    graph = assign_states(graph)
    return graph
        
def create_final_graph(nodes:int,edges:int)->nx.DiGraph:
    graph = create_random_graph(nodes,edges)
    graph = final_graph(graph)
    return graph


if __name__ == "__main__":
    graph = create_final_graph(10,0.1)
    nx.draw(graph,with_labels=True)
    plt.show()