import random
from statistics import mean
import tensorflow as tf
import keras
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from graph_cnn.graph.generate import create_final_graph
from keras.layers import Conv2D,GlobalAveragePooling2D,Flatten,Dense,Input,Add,Dropout,BatchNormalization,Activation,AveragePooling2D,LocallyConnected2D,MaxPooling2D


def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


def aux_layer(x, num_classes):
    """
    Auxiliary layer that performs global average pooling, flattening, and dense classification.

    Args:
        x (Tensor): Input tensor.
        num_classes (int): Number of classes for classification.

    Returns:
        Tensor: Output tensor after global average pooling, flattening, and dense classification.
    """
    global_avg_pool = GlobalAveragePooling2D()(x)
    flat = Flatten()(global_avg_pool)
    fc = Dense(units=num_classes, activation="softmax")(flat)
    return fc


def load_layer(layer):
    """
    Load a layer based on the given layer dictionary.

    Args:
        layer (dict): A dictionary containing the layer information.

    Returns:
        Layer: The loaded layer.

    Raises:
        ValueError: If the layer type is not supported.
    """
    if layer["layer_type"] == "Convolution":
        return Conv2D(
            filters=layer["filters"], kernel_size=layer["kernel_size"], padding="valid"
        )
    elif layer["layer_type"] == "MaxPooling":
        return MaxPooling2D(pool_size=layer["kernel_size"], strides=1)
    elif layer["layer_type"] == "AveragePooling2D":
        return AveragePooling2D(pool_size=layer["kernel_size"], strides=1)
    elif layer["layer_type"] == "LocallyConnected2D":
        return LocallyConnected2D(
            filters=layer["filters"], kernel_size=layer["kernel_size"], padding="valid"
        )
    raise ValueError(f"Unsupported layer type: {layer['layer_type']}")


def create_model(
    graph, input_shape=(224, 224, 3), num_classes=100, use_mean=True, include_aux=True
):
    """
    Create a graph convolutional neural network model.

    Args:
        graph (networkx.Graph): The graph structure defining the model architecture.
        input_shape (tuple): The shape of the input tensor (default: (224, 224, 3)).
        num_classes (int): The number of output classes (default: 100).
        use_mean (bool): Whether to use mean pooling for merging feature maps (default: True).
        include_aux (bool): Whether to include auxiliary layers in the model (default: True).

    Returns:
        tf.keras.Model: The created graph convolutional neural network model.
    """
    nodes = {}
    input_layer = Input(shape=input_shape)
    for node_ in graph.nodes():
        predecessors = list(graph.predecessors(node_))
        if not predecessors:
            filters = 32
            conv1 = Conv2D(
                filters=filters,
                kernel_size=graph.nodes[node_]["kernel_size"],
                activation=graph.nodes[node_]["activation"],
                padding="valid",
            )(input_layer)
            node_d = graph.nodes[node_]
            node_d["filters"] = filters
            current = load_layer(node_d)(input_layer)
            conv_shape = conv1.shape[1]
            current_shape = current.shape[1]
            kernel_size = abs(current_shape - conv_shape) + 1
            conv_dim = conv1.shape[-1]
            current_dim = current.shape[-1]
            filters = max(conv_dim, current_dim)
            node_d["kernel_size"] = (kernel_size, kernel_size)
            node_d["filters"] = filters
            current = load_layer(node_d)(current)
            add = Add()([conv1, current])
            normalized = BatchNormalization()(add)
            pool = MaxPooling2D()(normalized)
            activaton = Activation(graph.nodes[node_]["activation"])(pool)
            nodes[node_] = activaton
            print('inputs_built')
        else:
            if len(predecessors) > 1:
                if use_mean:
                    req_shape = int(mean([nodes[x].shape[-1] for x in predecessors]))
                else:
                    req_shape = max([nodes[x].shape[-1] for x in predecessors])
                req_dimension = max([nodes[x].shape[1] for x in predecessors])
                for predecessor in predecessors: 
                    kernel_size = nodes[predecessor].shape[1] - req_dimension + 1
                    if (
                        nodes[predecessor].shape[-1] != req_shape
                        or nodes[predecessor].shape[1] != req_dimension or kernel_size <= 0
                    ):
                        node = graph.nodes[predecessor]
                        node["filters"] = req_shape
                        node["kernel_size"] = (kernel_size, kernel_size)
                        layer = load_layer(node)(nodes[predecessor])
                        normalized = BatchNormalization()(layer)
                        activation = Activation(
                            node["activation"]
                        )(normalized)
                        nodes[predecessor] = activation
                concat = Add()([nodes[x] for x in predecessors])
            else:
                concat = nodes[predecessors[0]]
            nodes[node_] = concat
        print('predecessors_built')
        dropout_prob = random.uniform(0.2, 1)
        nodes[node_] = Dropout(dropout_prob)(nodes[node_])
    node_s = [nodes[node] for node in graph.nodes() if graph.out_degree(node) == 0]
    print(node_s)
    if use_mean:
        req_shape = mean([x.shape[-1] for x in node_s])
        req_dimension = max([x.shape[1] for x in node_s])
    else:
        req_shape = min([x.shape[-1] for x in node_s])
        req_dimension = max([x.shape[1] for x in node_s])
    print(req_shape, req_dimension)
    for node, _ in enumerate(node_s):
        kernel_size = node_s[node].shape[1] - req_dimension + 1
        if (
            node_s[node].shape[-1] != req_shape
            or node_s[node].shape[1] != req_dimension or kernel_size <= 0
        ):
            nodet = graph.nodes[node]
            nodet["filters"] = req_shape
            nodet["kernel_size"] = (kernel_size, kernel_size)
            print(kernel_size)
            layer = load_layer(nodet)(node_s[node])
            normalized = BatchNormalization()(layer)
            activation = Activation(graph.nodes[node]["activation"])(
                normalized
            )
            node_s[node] = activation
            print(node_s)
            add = Add()([activation, node_s[node]])
            nodes[node_] = add
    if include_aux:
        last_node = list(nodes.keys())[-1]
        aux = aux_layer(nodes[last_node], num_classes)
        model = keras.models.Model(inputs=input_layer, outputs=aux)
    else:
        model = keras.models.Model(inputs=input_layer, outputs=nodes[-1])
    return model


if __name__ == "__main__":
    set_seed(42)
    graph = create_final_graph(7, 0.1)
    nx.draw(graph, with_labels=True)
    plt.show()
    model = create_model(graph, use_mean=True, include_aux=True)
    tf.keras.utils.plot_model(
        model,
        to_file="model.png",
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
    )
    model.summary()
