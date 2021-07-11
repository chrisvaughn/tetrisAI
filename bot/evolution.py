import random
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class Genome:
    holes: float = 0
    roughness: float = 0
    lines: float = 0
    relative_height: float = 0
    absolute_height: float = 0
    cumulative_height: float = 0


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
                holes=random.uniform(-2, 2),
                roughness=random.uniform(-2, 2),
                lines=random.uniform(-2, 2),
                relative_height=random.uniform(-2, 2),
                absolute_height=random.uniform(-2, 2),
                cumulative_height=random.uniform(-2, 2),
            )
            genomes.append(genome)

        return genomes

    def select_best(self, genomes: List[Genome]):
        best_performers = []
        for i, genome in enumerate(genomes):
            lines = self.fitness(genome.__dict__)
            best_performers.append((i, lines))
        best_performers = sorted(best_performers, key=lambda x: x[1], reverse=True)
        return [(genomes[i], l) for i, l in best_performers[: self.select_best_n]]

    def combine_and_mutate(self, parents):
        children = []
        for i in range(self.population_size):
            mom_or_dad = [random.choice(parents)[0], random.choice(parents)[0]]
            child = Genome()
            for field in child.__dict__.keys():
                parent = random.choice(mom_or_dad)
                value = getattr(parent, field)
                if random.random() < self.mutation_rate:
                    value = (
                        value
                        + random.random() * self.mutation_step * 2
                        - self.mutation_step
                    )
                setattr(child, field, value)
            children.append(child)
        return children

    def run(self):
        genomes = self.create_initial()
        for gen in range(self.generations):
            print(f"Generation: {gen}")
            best = self.select_best(genomes)
            genomes = self.combine_and_mutate(best)
            print(best)
        return self.select_best(genomes)[0]
