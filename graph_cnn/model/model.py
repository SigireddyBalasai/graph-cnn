import tensorflow as tf
import numpy as np
import networkx as nx
import random
from matplotlib import pyplot as plt
from graph_cnn.graph.generate import create_final_graph
import tensorflow as tf

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

import tensorflow as tf

class AuxLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, trainable=True):
        super(AuxLayer, self).__init__()
        self.num_classes = num_classes
        self.layers_list = []

    def build(self, input_shape):
        current_shape = input_shape
        print(current_shape)
        while all(dim >= 5 for dim in current_shape[1:3]):
            kernels = current_shape[3]*2
            layer = tf.keras.layers.Conv2D(kernels, (3, 3), padding='same', activation='relu')
            self.layers_list.append(layer)
            current_shape = layer.compute_output_shape(current_shape)

            layer = tf.keras.layers.MaxPooling2D()
            self.layers_list.append(layer)
            current_shape = layer.compute_output_shape(current_shape)

        self.layers_list.append(tf.keras.layers.Flatten())
        self.layers_list.append(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

        super(AuxLayer, self).build(input_shape)

    def call(self, inputs):
        print(inputs)
        for layer in self.layers_list:
            inputs = layer(inputs)
        return inputs


def create_model(graph, input_shape=(224, 224, 3),num_classes=100):
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
    node_s = [nodes[node] for node in graph.nodes() if graph.out_degree(node) == 0]
    req_shape = min(node_s, key=lambda x: x.shape[-1]).shape[-1]
    print(req_shape)
    for node in range(len(node_s)):
        if node_s[node].shape[-1] != req_shape:
            nodes_ = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(1, 1))(node_s[node])
            nodes_ = tf.keras.layers.BatchNormalization()(nodes_)
            nodes_ = tf.keras.layers.Activation(graph.nodes[node]['activation'])(nodes_)
            nodes[node] = tf.keras.layers.Dropout(0.8)(nodes_)
    
    output_concat = tf.keras.layers.Add()(node_s)
    output_concat = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(3, 3))(output_concat)
    output_concat = tf.keras.layers.MaxPooling2D()(output_concat)
    output_concat = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=(3, 3))(output_concat)
    output_concat = tf.keras.layers.MaxPooling2D()(output_concat)
    output_concat = tf.keras.layers.Flatten()(output_concat)
    output_concat = tf.keras.layers.Dense(num_classes, activation='softmax')(output_concat)
    aux_layers = [AuxLayer(num_classes=num_classes)(nodes[node]) for node in nodes if random.uniform(0,1) > 0.5 and graph.in_degree(node) > 1]
    model = tf.keras.Model(inputs=input_layer, outputs=[output_concat,*aux_layers])
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