import random
from dataclasses import dataclass
from typing import Callable, List

from .evaluate import Weights


@dataclass
class Genome:
    weights: Weights = None
    fitness: float = 0.0


class GA:
    def __init__(
        self,
        population_size: int,
        generations: int,
        fitness: Callable,
    ):
        self.population_size = population_size
        self.generations = generations
        self.fitness = fitness

        self.select_best_n = 20
        self.mutation_rate = 0.05
        self.mutation_step = 0.2

    def create_initial(self) -> List[Genome]:
        genomes = []
        for i in range(self.population_size):
            genome = Genome(
                weights=Weights(
                    holes=random.uniform(-2, 2),
                    roughness=random.uniform(-2, 2),
                    lines=random.uniform(-2, 2),
                    relative_height=random.uniform(-2, 2),
                    absolute_height=random.uniform(-2, 2),
                    cumulative_height=random.uniform(-2, 2),
                    well_count=random.uniform(-2, 2),
                ),
                fitness=0.0,
            )
            genomes.append(genome)

        return genomes

    def select_best(self, genomes: List[Genome]):
        best_performers = []
        for genome in genomes:
            genome.fitness = self.fitness(genome.weights)
            best_performers.append(genome)
        best_performers = sorted(best_performers, key=lambda x: x.fitness, reverse=True)
        return best_performers[: self.select_best_n]

    def combine_and_mutate(self, parents: List[Genome]):
        children = []
        for i in range(self.population_size):
            mom_or_dad = [random.choice(parents), random.choice(parents)]
            child_weights = Weights()
            for field in child_weights.__dict__.keys():
                parent = random.choice(mom_or_dad)
                value = getattr(parent.weights, field)
                if random.random() < self.mutation_rate:
                    value = (
                        value
                        + random.random() * self.mutation_step * 2
                        - self.mutation_step
                    )
                setattr(child_weights, field, value)
            children.append(Genome(weights=child_weights))
        return children

    def run(self):
        genomes = self.create_initial()
        for gen in range(self.generations):
            print(f"Generation: {gen}")
            best = self.select_best(genomes)
            genomes = self.combine_and_mutate(best)
            print(best[0])
        return self.select_best(genomes)[0]
