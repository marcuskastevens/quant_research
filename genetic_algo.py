import sys
import random
import platform
from typing import List, Tuple

import ipdb
import pandas as pd
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        chromosome_length: int,
        mutation_rate: float,
        crossover_rate: float,
    ) -> None:

        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: List[List[int]] = self.initialize_population()

        self.fitness_evolution: List[float] = []

        return

    def initialize_population(self) -> List[List[int]]:
        return [
            [random.randint(0, 1) for _ in range(self.chromosome_length)]
            for _ in range(self.population_size)
        ]

    def fitness(self, chromosome: List[int]) -> int:

        # Define your fitness function here - currently the optimal solution is chromosome length
        return sum(chromosome)

    def selection(self) -> List[int]:

        # Randomly select two samples from the population
        tournament = random.sample(self.population, 2)

        # Return the sample that maximizes the self.fitness function
        return max(tournament, key=self.fitness)

    def crossover(
        self, parent1: List[int], parent2: List[int]
    ) -> Tuple[List[int], List[int]]:

        if random.random() < self.crossover_rate:

            # Must choose a number between index 1 and self.chromosome_length to ensure mating / reproduction / new offspring
            point = random.randint(1, self.chromosome_length - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]

        else:

            # If mutation probability threshold is violated, parents will not mate
            child1, child2 = parent1, parent2

        return child1, child2

    def mutate(self, chromosome: List[int]) -> List[int]:

        # Traverse through each gene
        for i in range(self.chromosome_length):

            # If mutation probability threshold is violated, mutate gene by flipping bit
            if random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]

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

        for _ in range(generations):

            self.evolve()
            best_chromosome = max(self.population, key=self.fitness)
            best_fitness = self.fitness(best_chromosome)
            self.fitness_evolution.append(best_fitness)
            print(f"Best Fitness: {best_fitness}")

            # This is an algo-specific condition to stop search early if optimallity is reached
            if best_fitness == self.chromosome_length:
                return

        return


def main() -> None:

    chromosome_length_frontier = [10, 50, 100, 500, 1000]

    for chromosome_length in chromosome_length_frontier:

        # Make population size a function of chromosome_length
        population_size = chromosome_length * 2

        # These are all parameters to tune via CV
        algo = GeneticAlgorithm(
            population_size=population_size,
            chromosome_length=chromosome_length,
            mutation_rate=0.001,
            crossover_rate=0.99,
        )
        algo.run(generations=1000)
        pd.Series(algo.fitness_evolution).plot(label="Genetic Algorithm")
        (pd.Series(algo.fitness_evolution) * 0 + algo.chromosome_length).plot(
            label="Optimal"
        )
        plt.legend()
        plt.show()

    return


# Example usage:
if __name__ == "__main__":
    main()
