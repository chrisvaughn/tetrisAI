import os
import pickle
import random
from dataclasses import dataclass
from typing import Callable, List

from progressbar import ETA, Bar, Percentage, ProgressBar

from .evaluate import Weights


@dataclass
class Genome:
    weights: Weights = None
    fitness: float = 0.0


@dataclass
class SaveState:
    genomes: List[Genome]
    current_generation: int


class GA:
    def __init__(
        self,
        population_size: int,
        generations: int,
        fitness: Callable,
        save_file: str,
    ):
        self.population_size = population_size
        self.generations = generations
        self.fitness = fitness
        self.save_file = save_file

        self.select_best_n = 15
        self.mutation_rate = 0.05
        self.mutation_step = 0.2

    def create_initial(self) -> List[Genome]:
        genomes = []
        for i in range(self.population_size):
            genome = Genome(
                weights=Weights(
                    holes=random.uniform(-1, 1),
                    roughness=random.uniform(-1, 1),
                    lines=random.uniform(-1, 1),
                    relative_height=random.uniform(-1, 1),
                    absolute_height=random.uniform(-1, 1),
                    cumulative_height=random.uniform(-1, 1),
                    well_count=random.uniform(-1, 1),
                    movements_required=random.uniform(-1, 1),
                ),
                fitness=0.0,
            )
            genomes.append(genome)

        return genomes

    def select_best(self, genomes: List[Genome], progress=True):
        pbar = None
        if progress:
            pbar = ProgressBar(
                widgets=[Percentage(), Bar(), ETA()], maxval=len(genomes)
            ).start()
        best_performers = []
        for i, genome in enumerate(genomes):
            genome.fitness = self.fitness(genome.weights)
            best_performers.append(genome)
            if pbar:
                pbar.update(i + 1)
        if pbar:
            pbar.finish()
        best_performers = sorted(best_performers, key=lambda x: x.fitness, reverse=True)
        return best_performers[: self.select_best_n]

    def combine_and_mutate(self, parents: List[Genome]):
        children = [parents[0], parents[1]]
        for i in range(self.population_size - 2):
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

    def run(self, resume: bool = False):
        if resume and os.path.isfile(self.save_file):
            with open(self.save_file, "rb") as f:
                save = pickle.load(f)
            genomes = save.genomes
            current = save.current_generation
        else:
            genomes = self.create_initial()
            current = 0
        for gen in range(current, self.generations):
            print(f"Generation: {gen}")
            best = self.select_best(genomes)
            genomes = self.combine_and_mutate(best)
            print(best[0])
            with open(self.save_file, "wb") as f:
                pickle.dump(SaveState(genomes, gen + 1), f)

        return self.select_best(genomes)[0]
