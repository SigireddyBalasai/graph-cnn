import tensorflow as tf
import numpy as np
import networkx as nx
import random
from matplotlib import pyplot as plt
from graph_cnn.graph.generate import create_final_graph

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def create_model(graph, input_shape=(224, 224, 3)):
    nodes = {}
    input_layer = tf.keras.layers.Input(shape=input_shape)

    for node in graph.nodes():
        predecessors = list(graph.predecessors(node))
        if not predecessors:
            if graph.nodes[node]['state'] == "Convolution":
                filters = 32
                nodes[node] = tf.keras.layers.Conv2D(filters=filters, kernel_size=graph.nodes[node]['kernel_size'], padding='same', activation=graph.nodes[node]['activation'])(input_layer)
            elif graph.nodes[node]['state'] == "MaxPooling":
                nodes[node] = tf.keras.layers.MaxPooling2D(pool_size=graph.nodes[node]['kernel_size'], strides=1, padding='same')(input_layer)
                nodes[node] = tf.keras.layers.Activation(graph.nodes[node]['activation'])(nodes[node])
            elif graph.nodes[node]['state'] == "AveragePooling":
                nodes[node] = tf.keras.layers.AveragePooling2D(pool_size=graph.nodes[node]['kernel_size'], strides=1, padding='same')(input_layer)
                nodes[node] = tf.keras.layers.Activation(graph.nodes[node]['activation'])(nodes[node])
        else:
            if len(predecessors) > 1:
                req_shape = min(list(map(lambda x: nodes[x],predecessors)), key=lambda x: x.shape[-1]).shape[-1]
                for predecessor in predecessors:
                    if nodes[predecessor].shape[-1] != req_shape:
                        nodes[predecessor] = tf.keras.layers.Conv2D(filters=req_shape, kernel_size=(1, 1))(nodes[predecessor])
                        nodes[predecessor] = tf.keras.layers.BatchNormalization()(nodes[predecessor])
                        nodes[predecessor] = tf.keras.layers.Activation(graph.nodes[predecessor]['activation'])(nodes[predecessor])
                        nodes[predecessor] = tf.keras.layers.Dropout(0.8)(nodes[predecessor])
                concat = tf.keras.layers.Add()([nodes[predecessor] for predecessor in predecessors])
            else:
                concat = nodes[predecessors[0]]
            if graph.nodes[node]['state'] == "Convolution":
                filters = concat.shape[-1] * 2
                nodes[node] = tf.keras.layers.Conv2D(filters=filters, kernel_size=graph.nodes[node]['kernel_size'], padding='same')(concat)
                nodes[node] = tf.keras.layers.BatchNormalization()(nodes[node]) 
                nodes[node] = tf.keras.layers.Activation(graph.nodes[node]['activation'])(nodes[node])
                nodes[node] = tf.keras.layers.Dropout(0.8)(nodes[node])
            elif graph.nodes[node]['state'] == "MaxPooling":
                nodes[node] = tf.keras.layers.MaxPooling2D(pool_size=graph.nodes[node]['kernel_size'], strides=1, padding='same')(concat)
                nodes[node] = tf.keras.layers.BatchNormalization()(nodes[node]) 
                nodes[node] = tf.keras.layers.Activation(graph.nodes[node]['activation'])(nodes[node])
            elif graph.nodes[node]['state'] == "AveragePooling":
                nodes[node] = tf.keras.layers.AveragePooling2D(pool_size=graph.nodes[node]['kernel_size'], strides=1, padding='same')(concat)
                nodes[node] = tf.keras.layers.BatchNormalization()(nodes[node])  # Add Batch Normalization
                nodes[node] = tf.keras.layers.Activation(graph.nodes[node]['activation'])(nodes[node])

        if graph.nodes[node]['state'] == "Convolution" or graph.nodes[node]['state'] == "MaxPooling" or graph.nodes[node]['state'] == "AveragePooling":
            dropout_prob = random.uniform(0.7, 1)
            nodes[node] = tf.keras.layers.Dropout(dropout_prob)(nodes[node])
    nodes = [nodes[node] for node in graph.nodes() if graph.out_degree(node) == 0]
    req_shape = min(nodes, key=lambda x: x.shape[-1]).shape[-1]
    print(req_shape)
    for node in range(len(nodes)):
            if nodes[node].shape[-1] != req_shape:
                nodes[node] = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(1, 1))(nodes[node])
                nodes[node] = tf.keras.layers.BatchNormalization()(nodes[node])
                nodes[node] = tf.keras.layers.Activation(graph.nodes[node]['activation'])(nodes[node])
                nodes[node] = tf.keras.layers.Dropout(0.8)(nodes[node])
    output_concat = tf.keras.layers.Concatenate()(nodes)
    model = tf.keras.Model(inputs=input_layer, outputs=output_concat)
    return model    


if __name__ == '__main__':
    set_seed(42)
    graph = create_final_graph(7, 0.4)
    nx.draw(graph, with_labels=True)
    print(graph, graph.nodes(data=True))
    plt.show()
    model = create_model(graph)
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, expand_nested=True)
    print(model.summary())