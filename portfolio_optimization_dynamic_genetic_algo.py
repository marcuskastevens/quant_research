"""
Long-Only at the moment
"""

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

# Fix random seed for alpha and variance calculation to directly benchmark algorithmic approaches
RANDOM_STATE = np.random.RandomState(20)  # np.random.RandomState(3)  # 0
N = 50
MU = 0.10
SIGMA = 0.20
ALPHA = RANDOM_STATE.normal(loc=MU, scale=SIGMA, size=N)
COV = np.diag(np.square(RANDOM_STATE.normal(loc=SIGMA, scale=SIGMA, size=N)))
W = np.dot(np.linalg.inv(COV), ALPHA) / np.linalg.norm(
    np.dot(np.linalg.inv(COV), ALPHA), 1
)
SR = np.dot(np.transpose(W), ALPHA) / np.sqrt(np.dot(np.dot(np.transpose(W), COV), W))
print(SR)


def min_max_scale(x):
    return (x - x.min()) / (x.max() - x.min())


class GeneticAlgorithm:

    def __init__(
        self,
        population_size: int,
        chromosome_length: int,
        mutation_rate: float,
        crossover_rate: float,
        population: list = None,
    ) -> None:

        # Define argument attributes
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Define method-generated attributes
        self.population: List[List[int]] = (
            population if population else self.initialize_population()
        )

        # Initialize attributes
        self.std_fitness_evolution: List[float] = []
        self.mean_fitness_evolution: List[float] = []
        self.best_fitness_evolution: List[float] = []
        self.best_chromosome_evolution: List[float] = []

        # Standard deviation for mutation distribution
        self.sigma = 1 / self.chromosome_length * 2

        return

    def initialize_population(self) -> List[List[int]]:
        """
        Implement custom initialization protocol.
        """

        # Randomly initialize and scale Standard Gaussian weights
        # initial_population = [
        #     np.random.standard_normal(size=self.chromosome_length)
        #     for _ in range(self.population_size)
        # ]
        # initial_population = [
        #     list(chromosome / np.sum(np.abs(chromosome)))
        #     for chromosome in initial_population
        # ]

        # NEW LOGIC - initialize long-only weights
        initial_population = [
            np.abs(np.random.standard_normal(size=self.chromosome_length))
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

        probabilities = np.random.random(self.chromosome_length)
        mutations = np.random.normal(
            loc=0, scale=self.sigma, size=self.chromosome_length
        )
        mutated_chromosome = list(
            np.abs((probabilities < self.mutation_rate) * mutations + chromosome)
            # (probabilities < self.mutation_rate) * mutations + chromosome, 0)  non-long-only
        )

        return mutated_chromosome

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

            # if i == generations * 0.75:
            #     self.sigma *= 5
            #     self.mutation_rate *= 5

            # if i == generations / 2:
            #     self.mutation_rate *= 2

            # Evolve generation
            self.evolve()

            # Evaluate each chromosome in population
            fitness = np.array(list(map(self.fitness, self.population)))
            best_chromosome = max(self.population, key=self.fitness)
            best_fitness = self.fitness(best_chromosome)

            # Update attributes
            self.best_fitness_evolution.append(best_fitness)
            self.std_fitness_evolution.append(np.std(fitness))
            self.mean_fitness_evolution.append(np.mean(fitness))
            self.best_chromosome_evolution.append(best_chromosome)

            if max(self.best_fitness_evolution) >= 0.999 * SR:
                break

        print(f"Best Fitness = {max(self.best_fitness_evolution)}")
        print(f"Residual = {max(self.best_fitness_evolution) - SR}")

        self.best_chromosome = max(self.best_chromosome_evolution, key=self.fitness)

        return


class ZigZagMutationAlgorithm(GeneticAlgorithm):
    """
    Algorithm = fluctuate between increasing and decreasing mutation crossover rate regimes. Inherits from GeneticAlgorithm.
    """

    def run(self, generations: int) -> None:

        n_regimes = 2  # int(np.sqrt(generations))
        n_evolutions_per_regime = int(
            generations
            / n_regimes
            / 2  # divide by two to make the evolution lenghts 25% of n_generations
        )
        increasing_regime = np.linspace(0, 1, n_evolutions_per_regime)
        decreasing_regime = np.linspace(1, 0, n_evolutions_per_regime)

        # Pre-allocate the mutation_schedule array with the exact needed length
        mutation_schedule = (
            np.zeros((n_regimes + 1) * n_evolutions_per_regime * 2)
            + self.mutation_rate  # initialize all mutation rates with original mutation rate
        )

        # Fill the mutation_schedule array
        for i in range(n_regimes + 1):
            if i % 2 == 0:
                mutation_schedule[
                    i * n_evolutions_per_regime : (i + 1) * n_evolutions_per_regime
                ] = increasing_regime  # decreasing_regime
            else:
                mutation_schedule[
                    i * n_evolutions_per_regime : (i + 1) * n_evolutions_per_regime
                ] = decreasing_regime  # increasing_regime

        # Truncate the mutation_schedule to the desired length
        mutation_schedule = mutation_schedule[:generations]

        for i in range(generations):

            # Evolve mutation and crossover rates
            self.mutation_rate = mutation_schedule[i]
            # self.crossover_rate = crossover_schedule[i]

            # Evolve generation
            self.evolve()

            # Evaluate each chromosome in population
            fitness = np.array(list(map(self.fitness, self.population)))
            best_chromosome = max(self.population, key=self.fitness)
            best_fitness = self.fitness(best_chromosome)

            # Update attributes
            self.best_fitness_evolution.append(best_fitness)
            self.std_fitness_evolution.append(np.std(fitness))
            self.mean_fitness_evolution.append(np.mean(fitness))
            self.best_chromosome_evolution.append(best_chromosome)

            if max(self.best_fitness_evolution) >= 0.999 * SR:
                break

        print(f"Best Fitness = {max(self.best_fitness_evolution)}")
        print(f"Residual = {max(self.best_fitness_evolution) - SR}")

        self.best_chromosome = max(self.best_chromosome_evolution, key=self.fitness)

        return


class ZigZagCrossoverAlgorithm(GeneticAlgorithm):
    """
    Algorithm = fluctuate between increasing and decreasing crossover rate regimes. Inherits from GeneticAlgorithm.
    """

    def run(self, generations: int) -> None:

        n_regimes = int(np.sqrt(generations))  # 10
        n_evolutions_per_regime = int(generations / n_regimes)
        increasing_regime = np.linspace(0, 1, n_evolutions_per_regime)
        decreasing_regime = np.linspace(1, 0, n_evolutions_per_regime)

        # Pre-allocate the crossover_schedule array with the exact needed length
        crossover_schedule = np.zeros((n_regimes + 1) * n_evolutions_per_regime)

        # Fill the crossover_schedule array
        for i in range(n_regimes + 1):
            if i % 2 == 0:
                crossover_schedule[
                    i * n_evolutions_per_regime : (i + 1) * n_evolutions_per_regime
                ] = decreasing_regime
            else:
                crossover_schedule[
                    i * n_evolutions_per_regime : (i + 1) * n_evolutions_per_regime
                ] = increasing_regime

        # Truncate the crossover_schedule to the desired length
        crossover_schedule = crossover_schedule[:generations]

        for i in range(generations):

            # Evolve mutation and crossover rates
            # self.mutation_rate = mutation_schedule[i]
            self.crossover_rate = crossover_schedule[i]

            # Evolve generation
            self.evolve()

            # Evaluate each chromosome in population
            fitness = np.array(list(map(self.fitness, self.population)))
            best_chromosome = max(self.population, key=self.fitness)
            best_fitness = self.fitness(best_chromosome)

            # Update attributes
            self.best_fitness_evolution.append(best_fitness)
            self.std_fitness_evolution.append(np.std(fitness))
            self.mean_fitness_evolution.append(np.mean(fitness))
            self.best_chromosome_evolution.append(best_chromosome)

            if max(self.best_fitness_evolution) >= 0.999 * SR:
                break

        print(f"Best Fitness = {max(self.best_fitness_evolution)}")
        print(f"Residual = {max(self.best_fitness_evolution) - SR}")

        self.best_chromosome = max(self.best_chromosome_evolution, key=self.fitness)

        return


class DMICAlgorithm(GeneticAlgorithm):
    """
    Decrease mutation rate / increase crossover rate algorithm. Inherits from GeneticAlgorithm.
    """

    def run(self, generations: int) -> None:

        # Evolve mutation rate according to the inverse of the square root function
        mutation_schedule = 1 / np.sqrt(np.arange(1, generations + 1))

        # Evolve crossover rate according to the min-max normalized log function
        crossover_schedule = min_max_scale(np.log(np.arange(1, generations + 1)))

        for i in range(generations):

            # Evolve mutation and crossover rates
            self.mutation_rate = mutation_schedule[i]
            self.crossover_rate = crossover_schedule[i]

            # Evolve generation
            self.evolve()

            # Evaluate each chromosome in population
            fitness = np.array(list(map(self.fitness, self.population)))
            best_chromosome = max(self.population, key=self.fitness)
            best_fitness = self.fitness(best_chromosome)

            # Update attributes
            self.best_fitness_evolution.append(best_fitness)
            self.std_fitness_evolution.append(np.std(fitness))
            self.mean_fitness_evolution.append(np.mean(fitness))
            self.best_chromosome_evolution.append(best_chromosome)

            if max(self.best_fitness_evolution) >= 0.999 * SR:
                break

        print(f"Best Fitness = {max(self.best_fitness_evolution)}")
        print(f"Residual = {max(self.best_fitness_evolution) - SR}")

        self.best_chromosome = max(self.best_chromosome_evolution, key=self.fitness)

        return


class IMDCAlgorithm(GeneticAlgorithm):
    """
    Increase mutation rate / decrease crossover rate algorithm. Inherits from GeneticAlgorithm.
    """

    def run(self, generations: int) -> None:

        # Evolve crossover rate according to the inverse of the square root function
        crossover_schedule = 1 / np.sqrt(np.arange(1, generations + 1))

        # Evolve mutation rate according to the min-max normalized log function
        mutation_schedule = min_max_scale(np.log(np.arange(1, generations + 1)))

        for i in range(generations):

            # Evolve mutation and crossover rates
            self.mutation_rate = mutation_schedule[i]
            self.crossover_rate = crossover_schedule[i]

            # Evolve generation
            self.evolve()

            # Evaluate each chromosome in population
            fitness = np.array(list(map(self.fitness, self.population)))
            best_chromosome = max(self.population, key=self.fitness)
            best_fitness = self.fitness(best_chromosome)

            # Update attributes
            self.best_fitness_evolution.append(best_fitness)
            self.std_fitness_evolution.append(np.std(fitness))
            self.mean_fitness_evolution.append(np.mean(fitness))
            self.best_chromosome_evolution.append(best_chromosome)

            if max(self.best_fitness_evolution) >= 0.999 * SR:
                break

        print(f"Best Fitness = {max(self.best_fitness_evolution)}")
        print(f"Residual = {max(self.best_fitness_evolution) - SR}")

        self.best_chromosome = max(self.best_chromosome_evolution, key=self.fitness)

        return


class DMAlgorithm(GeneticAlgorithm):
    """
    Decrease mutation rate / constant crossover rate algorithm. Inherits from GeneticAlgorithm.
    """

    def run(self, generations: int) -> None:

        # Evolve crossover rate according to the inverse of the square root function
        crossover_schedule = 1 / np.sqrt(np.arange(1, generations + 1))

        # Evolve mutation rate according to the min-max normalized log function
        mutation_schedule = min_max_scale(np.log(np.arange(1, generations + 1)))

        for i in range(generations):

            # Evolve mutation and crossover rates
            self.mutation_rate = mutation_schedule[i]
            self.crossover_rate = crossover_schedule[i]

            # Evolve generation
            self.evolve()

            # Evaluate each chromosome in population
            fitness = np.array(list(map(self.fitness, self.population)))
            best_chromosome = max(self.population, key=self.fitness)
            best_fitness = self.fitness(best_chromosome)

            # Update attributes
            self.best_fitness_evolution.append(best_fitness)
            self.std_fitness_evolution.append(np.std(fitness))
            self.mean_fitness_evolution.append(np.mean(fitness))
            self.best_chromosome_evolution.append(best_chromosome)

            if max(self.best_fitness_evolution) >= 0.999 * SR:
                break

        print(f"Best Fitness = {max(self.best_fitness_evolution)}")
        print(f"Residual = {max(self.best_fitness_evolution) - SR}")

        self.best_chromosome = max(self.best_chromosome_evolution, key=self.fitness)

        return


class ICAlgorithm(GeneticAlgorithm):
    """
    Increase crossover rate / constant mutation rate algorithm. Inherits from GeneticAlgorithm.
    """

    def run(self, generations: int) -> None:

        # Evolve crossover rate according to the inverse of the square root function
        crossover_schedule = 1 / np.sqrt(np.arange(1, generations + 1))

        # Evolve mutation rate according to the min-max normalized log function
        mutation_schedule = min_max_scale(np.log(np.arange(1, generations + 1)))

        for i in range(generations):

            # Evolve mutation and crossover rates
            self.mutation_rate = mutation_schedule[i]
            self.crossover_rate = crossover_schedule[i]

            # Evolve generation
            self.evolve()

            # Evaluate each chromosome in population
            fitness = np.array(list(map(self.fitness, self.population)))
            best_chromosome = max(self.population, key=self.fitness)
            best_fitness = self.fitness(best_chromosome)

            # Update attributes
            self.best_fitness_evolution.append(best_fitness)
            self.std_fitness_evolution.append(np.std(fitness))
            self.mean_fitness_evolution.append(np.mean(fitness))
            self.best_chromosome_evolution.append(best_chromosome)

            if max(self.best_fitness_evolution) >= 0.999 * SR:
                break

        print(f"Best Fitness = {max(self.best_fitness_evolution)}")
        print(f"Residual = {max(self.best_fitness_evolution) - SR}")

        self.best_chromosome = max(self.best_chromosome_evolution, key=self.fitness)

        return


class LinearAlgorithm(GeneticAlgorithm):
    """
    Linearly increase mutation rate / decrease crossover rate by 1 / generations algorithm. Inherits from GeneticAlgorithm.
    """

    def run(self, generations: int) -> None:

        for i in range(generations):

            # Linearly decrease mutation rate
            self.mutation_rate = 1 - i / generations

            # Linearly increase crossover rate
            self.crossover_rate = i / generations

            # Evolve generation
            self.evolve()

            # Evaluate each chromosome in population
            fitness = np.array(list(map(self.fitness, self.population)))
            best_chromosome = max(self.population, key=self.fitness)
            best_fitness = self.fitness(best_chromosome)

            # Update attributes
            self.best_fitness_evolution.append(best_fitness)
            self.std_fitness_evolution.append(np.std(fitness))
            self.mean_fitness_evolution.append(np.mean(fitness))
            self.best_chromosome_evolution.append(best_chromosome)

            if max(self.best_fitness_evolution) >= 0.999 * SR:
                break

        print(f"Best Fitness = {max(self.best_fitness_evolution)}")
        print(f"Residual = {max(self.best_fitness_evolution) - SR}")

        self.best_chromosome = max(self.best_chromosome_evolution, key=self.fitness)

        return


class LargeMutationAlgorithm(GeneticAlgorithm):
    """
    Simply increases the default mutation sigma parameter. Inherits from GeneticAlgorithm.

    A potential strategy with this algorithm is to evolve small populations concurrently to get fast convergnece
    while acheiving diversity in evolutionary paths.

    Since this has a large mutation magnitude, if ran in parallel with small populations,
    this evolutionary strategy should have maximal solution space coverage.

    From there, we then feed the optimal set of concurrently generated solutions into a single, less biased (less mutations),
    broader population to determine our final solution.
    """

    def run(self, generations: int) -> None:

        # Scale mutation sigma parameter
        self.sigma *= 10

        for i in range(generations):

            # Evolve generation
            self.evolve()

            # Evaluate each chromosome in population
            fitness = np.array(list(map(self.fitness, self.population)))
            best_chromosome = max(self.population, key=self.fitness)
            best_fitness = self.fitness(best_chromosome)

            # Update attributes
            self.best_fitness_evolution.append(best_fitness)
            self.std_fitness_evolution.append(np.std(fitness))
            self.mean_fitness_evolution.append(np.mean(fitness))
            self.best_chromosome_evolution.append(best_chromosome)

            if max(self.best_fitness_evolution) >= 0.999 * SR:
                break

        print(f"Best Fitness = {max(self.best_fitness_evolution)}")
        print(f"Residual = {max(self.best_fitness_evolution) - SR}")

        self.best_chromosome = max(self.best_chromosome_evolution, key=self.fitness)

        return


def main() -> None:

    # Define chromosome length
    chromosome_length = N

    # Define hyperparameters
    mutation_rate = 0.01  # 0.001
    crossover_rate = 0.85

    # Make population size a function of chromosome_length
    population_size = 10  # 75
    generations = 20000  # 10000

    # These are all parameters to tune via CV
    algo = ZigZagCrossoverAlgorithm(  # ZigZagCrossoverAlgorithm(  # ((  # ZigZagMutationAlgorithm( # GeneticAlgorithm(
        population_size=population_size,
        chromosome_length=chromosome_length,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
    )
    algo.run(generations=generations)
    pd.Series(algo.best_fitness_evolution).plot(label="Genetic Algorithm")
    (pd.Series(algo.best_fitness_evolution) * 0 + SR).plot(label="Optimal")
    plt.legend()
    plt.title("Best Fitness")
    plt.show()

    pd.Series(algo.mean_fitness_evolution).plot(label="Genetic Algorithm")
    (
        pd.Series(algo.mean_fitness_evolution)
        + 2 * pd.Series(algo.std_fitness_evolution)
    ).plot(label="+2STD")
    (
        pd.Series(algo.mean_fitness_evolution)
        - 2 * pd.Series(algo.std_fitness_evolution)
    ).plot(label="-2STD")
    # (pd.Series(algo.mean_fitness_evolution) * 0 + SR).plot(label="Optimal")
    plt.legend()
    plt.title("Mean Fitness")
    plt.show()

    print(algo.best_chromosome)

    # ------------------------------- Seed new algo with intelligently created population from previously run algorithm -------------------------------

    # Define hyperparameters
    mutation_rate = 0.005
    crossover_rate = 0.90

    # Increase population size for second algorithm
    population_size = 50
    generations = 10000

    algo = ZigZagCrossoverAlgorithm(
        population_size=population_size,
        chromosome_length=chromosome_length,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        population=algo.population,
    )
    algo.run(generations=generations)

    pd.Series(algo.best_fitness_evolution).plot(label="Genetic Algorithm")
    (pd.Series(algo.best_fitness_evolution) * 0 + SR).plot(label="Optimal")
    plt.legend()
    plt.title("Best Fitness - Intelligent Creator Initial Seed")
    plt.show()

    pd.Series(algo.mean_fitness_evolution).plot(label="Genetic Algorithm")
    (
        pd.Series(algo.mean_fitness_evolution)
        + 2 * pd.Series(algo.std_fitness_evolution)
    ).plot(label="+2STD")
    (
        pd.Series(algo.mean_fitness_evolution)
        - 2 * pd.Series(algo.std_fitness_evolution)
    ).plot(label="-2STD")
    # (pd.Series(algo.mean_fitness_evolution) * 0 + SR).plot(label="Optimal")
    plt.legend()
    plt.title("Mean Fitness - Intelligent Creator Initial Seed")
    plt.show()

    # ------------------------------- Seed new algo with intelligently created population from previously run algorithm -------------------------------

    # Define hyperparameters
    mutation_rate = 0.01  # 0.001
    crossover_rate = 0.90

    # Make population size a function of chromosome_length
    population_size = 100  # chromosome_length * 2
    generations = 2500

    # These are all parameters to tune via CV
    algo = LargeMutationAlgorithm(  # GeneticAlgorithm(
        population_size=population_size,
        chromosome_length=chromosome_length,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        population=algo.population,
    )
    algo.run(generations=generations)

    pd.Series(algo.best_fitness_evolution).plot(label="Genetic Algorithm")
    (pd.Series(algo.best_fitness_evolution) * 0 + SR).plot(label="Optimal")
    plt.legend()
    plt.title("Best Fitness - Intelligent Creator Initial Seed")
    plt.show()

    pd.Series(algo.mean_fitness_evolution).plot(label="Genetic Algorithm")
    (
        pd.Series(algo.mean_fitness_evolution)
        + 2 * pd.Series(algo.std_fitness_evolution)
    ).plot(label="+2STD")
    (
        pd.Series(algo.mean_fitness_evolution)
        - 2 * pd.Series(algo.std_fitness_evolution)
    ).plot(label="-2STD")
    # (pd.Series(algo.mean_fitness_evolution) * 0 + SR).plot(label="Optimal")
    plt.legend()
    plt.title("Mean Fitness - Intelligent Creator Initial Seed")
    plt.show()

    # ------------------------------- Seed new algo with intelligently created population from previously run algorithm -------------------------------

    # Define hyperparameters
    mutation_rate = 0.005
    crossover_rate = 0.99

    # Increase population size for second algorithm
    population_size = 75
    generations = 10000

    algo = ZigZagCrossoverAlgorithm(  # GeneticAlgorithm(  # LargeMutationAlgorithm(  # DMAlgorithm(  # LinearAlgorithm
        population_size=population_size,
        chromosome_length=chromosome_length,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        population=algo.population,
    )
    algo.run(generations=generations)

    pd.Series(algo.best_fitness_evolution).plot(label="Genetic Algorithm")
    (pd.Series(algo.best_fitness_evolution) * 0 + SR).plot(label="Optimal")
    plt.legend()
    plt.title("Best Fitness - Intelligent Creator Initial Seed")
    plt.show()

    pd.Series(algo.mean_fitness_evolution).plot(label="Genetic Algorithm")
    (
        pd.Series(algo.mean_fitness_evolution)
        + 2 * pd.Series(algo.std_fitness_evolution)
    ).plot(label="+2STD")
    (
        pd.Series(algo.mean_fitness_evolution)
        - 2 * pd.Series(algo.std_fitness_evolution)
    ).plot(label="-2STD")
    plt.legend()
    plt.title("Mean Fitness - Intelligent Creator Initial Seed")
    plt.show()

    return


# Example usage:
if __name__ == "__main__":
    main()
