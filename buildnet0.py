# Description: This file contains the implementation of the genetic algorithm
# noga ben-ari 208304220
# shahar oron 322807231
import copy
from itertools import tee
import random
import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt
import pickle


# activation functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x):
    return np.maximum(0.1 * x, x)


population_size = 150
generations_num = 500
mutation_rate = 0.3
selection_rate = 0.1
input_size = 16
layer1_size = 32
layer2_size = 16
output_size = 1
activation_f = leaky_relu
npairs = 3
untoched = 0.3


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def f(row: str):
    return [x for x in row[0]]


# organize the data
def split_data():
    train_file = input("please enter train file:")
    test_file = input("please enter test file:")
    # train_file = "tr.txt"
    # test_file = "te.txt"

    # for train
    read_train = pd.read_csv(train_file, sep='\s+', header=None, dtype=str)
    training_examples = read_train.iloc[:, 0].astype(str)
    transpose = training_examples.to_frame()
    train_matrix = transpose.apply(f, axis='columns', result_type='expand')
    train_matrix = train_matrix.to_numpy(int)
    training_labels = read_train.iloc[:, 1].values.astype(int)
    # for test
    read_test = pd.read_csv(test_file, sep='\s+', header=None, dtype=str)
    test_examples = read_test.iloc[:, 0].astype(str)
    transpose = test_examples.to_frame()
    test_matrix = transpose.apply(f, axis='columns', result_type='expand')
    test_matrix = test_matrix.to_numpy(int)
    test_labels = read_test.iloc[:, 1].values.astype(int)


    training_examples = train_matrix[:train_matrix.shape[0]]
    test_examples = test_matrix[:test_matrix.shape[0]]
    return training_examples, training_labels, test_examples, test_labels


# initialize a neural network
def initiate_nn():
    nn_model = Individual()
    nn_model.add_weight(input_size, layer1_size)
    nn_model.add_weight(layer1_size, layer2_size)
    nn_model.add_weight(layer2_size, output_size)
    return nn_model


# accuracy calculation
def calculate_accuracy(train_labels, predictions):
    return np.mean(np.where((predictions == train_labels) > 0.5, 1, 0))


class GeneticAlg:
    def __init__(self):
        self.population = self.init_population()
        self.x_train, self.y_train, self.x_test, self.y_test = split_data()
        self.best_fitness = 0
        self.same_fitness_count = 0
        self.generation = 0
        self.generations = []
        self.list_of_accuracy = []

    # Rank Selection
    def rank_selection(self, population):
        fitness_scores = [self.calculate_fitness(network) for network in population]
        sum_fitness = sum(fitness_scores)
        probabilities = [score / sum_fitness for score in fitness_scores]
        selected_parents = np.random.choice(population, size=2, replace=False, p=probabilities)
        return selected_parents


    # initiation
    def init_population(self):
        population = []
        for _ in range(population_size):
            network = initiate_nn()
            population.append(network)
        return population

    # selection
    def selection(self, fitness_list):
        sorted_indices = np.argsort(fitness_list)[::-1]
        top_individuals = [self.population[i] for i in sorted_indices[:int(population_size * selection_rate)]]
        remaine_individuals = list(set(self.population) - set(top_individuals))
        return top_individuals, remaine_individuals

    # crossover
    def crossover(self, selected_parents, top_individuals):
        offspring = []
        for _ in range((population_size - len(top_individuals)) // 2):
            parent1 = np.random.choice(selected_parents)
            parent2 = np.random.choice(top_individuals)
            offspring1, offspring2 = parent1.crossover(parent2)
            offspring.append(offspring1)
            offspring.append(offspring2)
        return offspring

    # fitness calculation
    def calculate_fitness(self, network):
        predictions = network.predict(self.x_train)
        return calculate_accuracy(self.y_train, predictions)

    # the flow of the genetic algorithm
    def run_algo(self):
        for i in range(generations_num):
            self.generation += 1
            self.generations.append(i + 1)

            # calculate the fitness for the population
            fitness_scores = []
            for network in self.population:
                fitness = self.calculate_fitness(network)
                fitness_scores.append(round(fitness, 5))

            # save the highest fitness
            current_fitness = max(fitness_scores)
            print(f"Generation {i + 1} best fitness score: {max(fitness_scores)}")
            self.list_of_accuracy.append(max(fitness_scores))

            # convergence
            if current_fitness > self.best_fitness:
                self.best_fitnes = current_fitness
                self.same_fitness_count = 0
            else:
                # if it was the same fitness score
                self.same_fitness_count += 1

            if self.same_fitness_count >= 30:
                # if the fitness was the same 30 times, stop the running
                print("the algorithm has converged, stopping the running")
                break

            top_individuals, remaine_individuals = self.selection(fitness_scores)

            selected_parents = self.rank_selection(remaine_individuals)

            # crossover
            offsprings = self.crossover(selected_parents, top_individuals)

            # saves some individuals that won't be mutated to the next generation
            num_untouched = int((population_size - len(top_individuals)) * untoched)
            untouched = offsprings[:num_untouched]

            # mutate the other offsprings, that can be touched
            for offspring in offsprings[num_untouched:]:
                offspring.mutation()

            # the creation of the next population
            self.population = top_individuals + untouched + offsprings[num_untouched:]

            if self.same_fitness_count > 3:
                # lamarck version
                print("lamarckian is on")
                new_population = []
                for network in self.population:
                    new_population.append(self.lamark_alg(network))
                self.population = new_population

        fitness_scores = [self.calculate_fitness(network) for network in self.population]
        best_ind = self.population[np.argmax(fitness_scores)]
        return best_ind

    # lamrck implementation
    def lamark_alg(self, network):
        print("lamarkian is on")
        old_fitness = self.calculate_fitness(network)  # saves the old fitness
        new_network = copy.deepcopy(network)
        for i in range(npairs):
            layer_num = np.random.choice(len(network.layers))
            # choose a random weight in the layer
            layer_weights = network.layers[layer_num]
            rand_idx = np.random.randint(0, layer_weights.shape[0])
            rand_weight = np.random.randn(layer_weights.shape[1])

            # replace the random weight with the new random weight
            network.layers[layer_num][rand_idx] = rand_weight

        new_fitness = self.calculate_fitness(new_network)
        if new_fitness > old_fitness:
            return new_network
        else:
            return network


class Individual:

    def __init__(self):
        self.layers = []

    # add layers to the nn
    def add_layer(self, layer):
        self.layers.append(layer)

    # ass weights to the layers
    def add_weight(self, inside, outside):
        self.add_layer(np.random.randn(inside, outside) * np.sqrt(1 / inside))

    def forward(self, examples: np.ndarray) -> np.ndarray:
        # input to the layers
        input = examples
        for w in self.layers[:-1]:
            input = np.dot(input, w)
        # last layer
        w = self.layers[-1]
        input = np.dot(input, w)
        input = activation_f(input)
        return input

    def predict(self, inputs):
        outputs = self.forward(inputs)
        binary_predictions = (outputs > 0.5).astype(int)
        return binary_predictions.flatten()

    # crossover
    def crossover(self, other_network):
        child1 = Individual()
        child2 = Individual()
        # Get the layers from the parents
        for w1, w2 in zip(self.layers, other_network.layers):
            # Perform the crossover operation
            columns = np.random.choice([True, False], size=w1.shape[1])
            c1w = np.where(columns, w1, w2)
            c2w = np.where(columns, w2, w1)

            # Create new child individuals with the updated layers and fitness score of 0
            child1.layers.append(c1w)
            child2.layers.append(c2w)

        return child1, child2

    # mutation
    def mutation(self):

        if np.random.random() < mutation_rate:
            # choose a random layer from individual.layers
            layer_num = np.random.choice(len(self.layers))
            # choose a random weight in the layer
            layer_weights = self.layers[layer_num]
            rand_idx = np.random.randint(0, layer_weights.shape[0])
            rand_weight = np.random.randn(layer_weights.shape[1])

            # replace the random weight with the new random weight
            self.layers[layer_num][rand_idx] = rand_weight


# Creates a PNG image with a line plot of the number of infected cells over time (generations).
def create_png(generations, accuracy):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    ax.plot(generations, accuracy)
    ax.set_xlabel("Generations")
    ax.set_ylabel("accuracy")
    ax.set_title("Number of fitness score Over Time")
    plt.savefig("fitness_score_over_time.png")


#

def main():
    np.seterr(all="raise")
    start_time = time.perf_counter()
    ga = GeneticAlg()
    # train
    best_network = ga.run_algo()
    # Open the file in write mode
    with open('wnet.txt', 'w') as file:
        # Save the layers
        file.write("Layers:\n")
        for layer in best_network.layers:
            file.write(str(layer) + "\n")

        # Save the weights
        file.write("\nWeights:\n")
        for layer in best_network.layers:
            if isinstance(layer, tuple):
                inside, outside = layer
                weights = np.random.randn(inside, outside) * np.sqrt(1 / inside)
                file.write("Layer Weights:\n")
                for weight_row in weights:
                    file.write(" ".join(str(weight) for weight in weight_row) + "\n")
                file.write("\n")
    # test
    predict_test = best_network.predict(ga.x_test)
    accuracy = calculate_accuracy(ga.y_test, predict_test)
    print(f"accuracy: {accuracy}")

    create_png(ga.generations, ga.list_of_accuracy)
    end_time = time.perf_counter()
    total_time = (end_time - start_time) / 60
    print("running time:", total_time)


if __name__ == '__main__':
    main()