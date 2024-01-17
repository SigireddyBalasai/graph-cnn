import tensorflow as tf
import numpy as np
import networkx as nx
import random
from matplotlib import pyplot as plt
from graph_cnn.graph.generate import create_final_graph
from statistics import mean

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

class AuxLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(AuxLayer, self).__init__()
        self.num_classes = num_classes
        self.layers_list = []

    def build(self, input_shape):
        current_shape = input_shape
        print(current_shape)
        self.layers_list.append(tf.keras.layers.GlobalAveragePooling2D())
        self.layers_list.append(tf.keras.layers.Dense(units=self.num_classes, activation='softmax'))
    def call(self, inputs):
        for layer in self.layers_list:
            inputs = layer(inputs)
        return inputs


def create_model(graph, input_shape=(224, 224, 3), num_classes=100):
    nodes = {}
    input_layer = tf.keras.layers.Input(shape=input_shape)
    for node in graph.nodes():
        predecessors = list(graph.predecessors(node))
        if not predecessors:
            if graph.nodes[node]['state'] == "Convolution":
                filters = 32
                nodes[node] = tf.keras.layers.Conv2D(filters=filters, kernel_size=graph.nodes[node]['kernel_size'], activation=graph.nodes[node]['activation'], padding='valid')(input_layer)
            elif graph.nodes[node]['state'] == "MaxPooling":
                nodes[node] = tf.keras.layers.MaxPooling2D(pool_size=graph.nodes[node]['kernel_size'], padding='valid')(input_layer)
                nodes[node] = tf.keras.layers.Activation(graph.nodes[node]['activation'])(nodes[node])
            elif graph.nodes[node]['state'] == "AveragePooling":
                nodes[node] = tf.keras.layers.AveragePooling2D(pool_size=graph.nodes[node]['kernel_size'], padding='valid')(input_layer)
                nodes[node] = tf.keras.layers.Activation(graph.nodes[node]['activation'])(nodes[node])
        else:
            if len(predecessors) > 1:
                req_shape = min([nodes[x].shape[-1] for x in predecessors])
                req_dimension = min([nodes[x].shape[1] for x in predecessors])
                for predecessor in predecessors:
                    kernel_size = nodes[predecessor].shape[1] - req_dimension + 1
                    if nodes[predecessor].shape[-1] != req_shape or nodes[predecessor].shape[1] != req_dimension:
                        nodes[predecessor] = tf.keras.layers.Conv2D(filters=req_shape, kernel_size=(kernel_size, kernel_size), padding='valid')(nodes[predecessor])
                        nodes[predecessor] = tf.keras.layers.Activation(graph.nodes[predecessor]['activation'])(nodes[predecessor])
                        nodes[predecessor] = tf.keras.layers.BatchNormalization()(nodes[predecessor])  # Do Batch Normalization after Activation
                        nodes[predecessor] = tf.keras.layers.Dropout(0.2)(nodes[predecessor])
                concat = tf.keras.layers.Add()([nodes[predecessor] for predecessor in predecessors])
                concat = tf.keras.layers.Conv2D(filters=concat.shape[-1]*2, kernel_size=(1, 1), padding='valid')(concat)
            else:
                concat = nodes[predecessors[0]]
            if concat.shape[1] - graph.nodes[node]['kernel_size'][0] + 1 > 5:
                if graph.nodes[node]['state'] == "Convolution":
                    filters = concat.shape[-1] * 2
                    nodes[node] = tf.keras.layers.Conv2D(filters=filters, kernel_size=graph.nodes[node]['kernel_size'], padding='valid')(concat)
                    nodes[node] = tf.keras.layers.Activation(graph.nodes[node]['activation'])(nodes[node])
                    nodes[node] = tf.keras.layers.BatchNormalization()(nodes[node])  # Do Batch Normalization after Activation
                    nodes[node] = tf.keras.layers.Dropout(random.uniform(0, 1))(nodes[node])
                elif graph.nodes[node]['state'] == "MaxPooling":
                    nodes[node] = tf.keras.layers.MaxPooling2D(pool_size=graph.nodes[node]['kernel_size'], padding='valid')(concat)
                    nodes[node] = tf.keras.layers.Activation(graph.nodes[node]['activation'])(nodes[node])
                    nodes[node] = tf.keras.layers.BatchNormalization()(nodes[node])
                elif graph.nodes[node]['state'] == "AveragePooling":
                    nodes[node] = tf.keras.layers.AveragePooling2D(pool_size=graph.nodes[node]['kernel_size'], padding='valid')(concat)
                    nodes[node] = tf.keras.layers.Activation(graph.nodes[node]['activation'])(nodes[node])
                    nodes[node] = tf.keras.layers.BatchNormalization()(nodes[node])  # Do Batch Normalization after Activation
            else:
                nodes[node] = concat
        if graph.nodes[node]['state'] in ["Convolution", "MaxPooling", "AveragePooling"]:
            dropout_prob = random.uniform(0.7, 1)
            nodes[node] = tf.keras.layers.Dropout(dropout_prob)(nodes[node])
    node_s = [nodes[node] for node in graph.nodes() if graph.out_degree(node) == 0]
    req_shape = mean([x.shape[-1] for x in node_s])
    req_dimension = min([x.shape[1] for x in node_s])
    for node in range(len(node_s)):
        kernel_size = node_s[node].shape[1] - req_dimension + 1
        if node_s[node].shape[-1] != req_shape or node_s[node].shape[1] != req_dimension:
            nodes_ = tf.keras.layers.Conv2D(filters=req_shape, kernel_size=(kernel_size, kernel_size), padding='valid')(node_s[node])
            nodes_ = tf.keras.layers.Activation(graph.nodes[node]['activation'])(nodes_)
            nodes_ = tf.keras.layers.BatchNormalization()(nodes_)  # Do Batch Normalization after Activation
            node_s[node] = tf.keras.layers.Dropout(0.2)(nodes_)

    output_concat = tf.keras.layers.Add()(node_s)
    output_concat = AuxLayer(num_classes=num_classes)(output_concat)
    aux_layers = [AuxLayer(num_classes=num_classes)(nodes[node]) for node in nodes if random.uniform(0, 1) > 0.5 and graph.in_degree(node) > 1]
    model = tf.keras.Model(inputs=input_layer, outputs=[output_concat, *aux_layers])
    return model


if __name__ == '__main__':
    set_seed(42)
    graph = create_final_graph(7, 0.4)
    nx.draw(graph, with_labels=True)
    plt.show()
    model = create_model(graph)
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, expand_nested=True)
    model.summary()
