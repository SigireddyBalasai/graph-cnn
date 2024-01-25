from graph_cnn.individual import Individual
import random
import os
import tensorflow as tf

class Generation:
    def __init__(self, input_size, output_size, nodes, edges, population, limit, train_ds, test_ds,loss,optimizer,metrics,callbacks):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = nodes
        self.edges = edges
        self.population = population
        self.limit = limit
        self.generation = 0
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.callbacks = callbacks
        self.create_population()

    def create_population(self):
        self.population = [Individual(self.input_size, self.output_size, self.nodes, self.edges) for _ in range(self.population)]
    
    def score_population(self):
        population = self.population
        for individual in population:
            individual.evaluate(test_ds=self.test_ds)
            print("individual score: ", individual.get_score())
        population.sort(key=lambda x: x.get_score(), reverse=True)
        self.population = population
    
    def get_best_individual(self):
        return self.population[0]

    def mutate_population(self,mutation_rate):
        mutated = []
        for individual in self.population:
            if random.random() < mutation_rate:
                try:
                    mutated.append(individual.mutate())
                    print(f"Mutating individual with score: {individual.get_score()}")
                except:
                    pass
        self.population += mutated
    
    def crossover_population(self, crossover_rate):
        crossed = []
        for i in range(len(self.population)):
            if random.random() < crossover_rate:
                child1 , child2 = self.population[i].crossover(self.population[i-1])
                crossed.append(child1)
                crossed.append(child2)
        self.population += crossed
        return self.population
    
    def next_generation(self, mutation_rate, crossover_rate):
        self.score_population()
        self.mutate_population(mutation_rate)
        #self.crossover_population(crossover_rate)
        self.population.sort(key=lambda x: x.get_score(), reverse=True)
        return self.population
    
    def run(self, n, mutation_rate, crossover_rate):
        for i in range(n):
            os.makedirs(f'generation_{self.generation}', exist_ok=True)
            print(f'Generation: {self.generation}')
            self.next_generation(mutation_rate, crossover_rate)
            print(f'Best score: {self.population[0].get_score()}')
            model = self.population[0].get_model()
            model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            model.summary()
            model.fit(self.train_ds, epochs=15, validation_data=self.test_ds,callbacks=self.callbacks)
            print('---------------------------------------------------')
            for individual in self.population:
                print("individual score: ", individual.get_score())
                #print("individual architecture: ", individual.get_model().summary())
                individual.save_model(f'generation_{self.generation}')
            self.generation += 1
