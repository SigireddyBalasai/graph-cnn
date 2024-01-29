import networkx as nx
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from graph_cnn.graph.generate import (
    create_final_graph,
    create_random_graph,
    assign_states,
)
from graph_cnn.graph.generate import cross_over as crossover
from graph_cnn.graph.generate import mutate as mutate_dag
from graph_cnn.model import create_model
import threading
import threading


class Individual:
    def __init__(self, input_size, output_size, nodes, edges):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = nodes
        self.edges = edges
        self.graph = self.create_random_graph()
        self.model = create_model(
            self.graph, input_shape=self.input_size, num_classes=self.output_size
        )
        self.score = 0

    def create_random_graph(self):
        return create_final_graph(self.nodes, self.edges)

    def evaluate(self, test_ds):
        train_ds = list(test_ds)
        print("evaluating")
        results = []
        threads = []

        def evaluate_single_thread(data):
            result = self.evaluate_single(data)
            results.append(result)

        for data in train_ds:
            thread = threading.Thread(target=evaluate_single_thread, args=(data,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        accuracies = [result[-1] for result in results]
        accuracies = np.array(accuracies)
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        score = mean_accuracy / std_accuracy
        self.score = score

    def evaluate_single(self, data):
        loss = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=tf.keras.metrics.CategoricalAccuracy(),
        )
        ans = self.model.evaluate(data[0], data[1], verbose=0)
        return ans

    def mutate(self):
        self.graph = mutate_dag(self.graph)
        self.model = create_model(self.graph, self.input_size, self.output_size)
        return self

    def crossover(self, other):
        self.graph = crossover(self.graph, other.graph)
        self.model = create_model(self.graph, self.input_size, self.output_size)
        return self

    def get_model(self):
        return self.model

    def get_graph(self):
        plt.figure(figsize=(10, 10))
        nx.draw(self.graph, with_labels=True)
        plt.draw()
        plt.show()

    def save_model(self, folder):
        model = tf.keras.models.clone_model(self.model)
        print("saving model")
        score = self.score
        print(score, "score")
        tf.keras.utils.plot_model(
            self.model, to_file=f"{folder}/{self.score}.png", show_shapes=True
        )
        print(f"image saved in {folder}/{self.score}.png")
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        model.save(f"{folder}/{self.score}")
        self.model = model

    def get_score(self):
        return self.score
