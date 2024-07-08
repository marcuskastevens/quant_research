import sys
import random
import platform
from typing import List, Tuple

import ipdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
TODO: Incoporate off-diagonal solutions.
"""

N = 50
MU = 0.10
SIGMA = 0.90
ALPHA = np.random.normal(loc=MU, scale=SIGMA, size=N)
COV = np.diag(np.square(np.random.normal(loc=SIGMA, scale=SIGMA, size=N)))
W = np.dot(np.linalg.inv(COV), ALPHA) / np.linalg.norm(
    np.dot(np.linalg.inv(COV), ALPHA), 1
)
SR = np.dot(np.transpose(W), ALPHA) / np.sqrt(np.dot(np.dot(np.transpose(W), COV), W))
print(SR)


class GeneticAlgorithm:

    def __init__(
        self,
        population_size: int,
        chromosome_length: int,
        mutation_rate: float,
        crossover_rate: float,
    ) -> None:

        # Define argument attributes
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Define method-generated attributes
        self.population: List[List[int]] = self.initialize_population()

        # Initialize attributes
        self.fitness_evolution: List[float] = []
        self.mean_fitness_evolution: List[float] = []

        return

    def initialize_population(self) -> List[List[int]]:
        """
        Implement custom initialization protocol.
        """

        # Randomly initialize and scale Standard Gaussian weights
        initial_population = [
            np.random.standard_normal(size=self.chromosome_length)
            for _ in range(self.population_size)
        ]
        initial_population = [
            list(chromosome / np.sum(np.abs(chromosome)))
            for chromosome in initial_population
        ]

        return initial_population

    def fitness(self, chromosome: List[int]) -> int:

        # Ex ante Sharpe Ratio - chromosomes represent portfolio weights
        sr = np.dot(np.transpose(chromosome), ALPHA) / np.sqrt(
            np.dot(np.dot(np.transpose(chromosome), COV), chromosome)
        )

        # Define your fitness function here
        return sr

    def selection(self) -> List[int]:
        """
        This selection criterion randomly selects combatants, but intelligently filters for the fittest combatant
        """

        # Select 10% of the population - this is a paremeter that will need to be tuned
        # n_combatants = int(0.10 * self.population_size)

        # Randomly select two samples from the population
        tournament = random.sample(self.population, 2)

        # Return the sample that maximizes the self.fitness function
        return max(tournament, key=self.fitness)

    def crossover(
        self, parent1: List[int], parent2: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Crosses genes between two strong parents (ensure strength due to selection protocol) to potentially make an even stronger child.
        """

        if random.random() < self.crossover_rate:

            # Must choose a number between index 1 and self.chromosome_length to ensure mating / reproduction / new offspring
            index = random.randint(1, self.chromosome_length - 1)
            child1 = parent1[:index] + parent2[index:]
            child2 = parent2[:index] + parent1[index:]

        else:

            # If crossover threshold is not violated, parents will not mate
            child1, child2 = parent1, parent2

        return child1, child2

    def mutate(self, chromosome: List[int]) -> List[int]:
        """
        Mutation protocol will differ per implementation. Mutation rate should be lower enough such that good chromosomes are not completely altered.
        """

        # Traverse through each gene
        for i in range(self.chromosome_length):

            # If mutation probability threshold is violated, mutate gene by flipping bit
            if random.random() < self.mutation_rate:

                # Adjust this perturbation based on derivative between current chromose and prev best chromosome

                # Mutation protocol to randomly perturb portfolio weights based on Gaussian noise - standard deviation is a function of portfolio size
                sigma = 1 / self.chromosome_length
                mutation = np.random.normal(loc=0, scale=sigma)
                chromosome[i] += mutation

        return chromosome

    def evolve(self) -> None:

        # Initialize new population that will later overwrite the current population
        new_population: List[List[int]] = []

        # Iterate over half of the population size since each iteration creates two offspring, thereby maintaining the population size across generations
        for _ in range(self.population_size // 2):

            # Intelligently, but randomly select two parents from the population
            parent1 = self.selection()
            parent2 = self.selection()

            # Randomly mate two parents
            child1, child2 = self.crossover(parent1, parent2)

            # Randomly mutate offspring and update fill population
            new_population.extend([self.mutate(child1), self.mutate(child2)])

        # Once new population is filled with new children, overwrite new population
        self.population = new_population

    def run(self, generations: int) -> None:

        for i in range(generations):

            self.evolve()
            best_chromosome = max(self.population, key=self.fitness)
            best_fitness = self.fitness(best_chromosome)
            self.fitness_evolution.append(best_fitness)
            self.mean_fitness_evolution.append(
                np.mean(list(map(self.fitness, self.population)))
            )
            print(f"Best Fitness {i}: {best_fitness}")

            if best_fitness >= 0.999 * SR:
                break

        self.best_chromosome = best_chromosome

        return


def main() -> None:

    # Define hyperparameters
    mutation_rate = 0.01
    crossover_rate = 0.80

    # Define chromosome length
    chromosome_length = len(ALPHA)

    # Make population size a function of chromosome_length
    population_size = chromosome_length * 2

    # These are all parameters to tune via CV
    algo = GeneticAlgorithm(
        population_size=population_size,
        chromosome_length=chromosome_length,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
    )
    algo.run(generations=10000)
    pd.Series(algo.fitness_evolution).plot(label="Genetic Algorithm")
    (pd.Series(algo.fitness_evolution) * 0 + SR).plot(label="Optimal")
    plt.legend()
    plt.title("Best Fitness")
    plt.show()

    pd.Series(algo.mean_fitness_evolution).plot(label="Genetic Algorithm")
    (pd.Series(algo.fitness_evolution) * 0 + SR).plot(label="Optimal")
    plt.legend()
    plt.title("Mean Fitness")
    plt.show()

    print(algo.best_chromosome)

    return


# Example usage:
if __name__ == "__main__":
    main()
