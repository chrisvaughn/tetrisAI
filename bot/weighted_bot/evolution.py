import multiprocessing
import os
import pickle
import random
from dataclasses import dataclass
from typing import Callable, List

from progressbar import ETA, Bar, Percentage, ProgressBar

from .evaluate import Weights
from .evaluation_pool import _evaluate_genome, _genome_worker_init


@dataclass
class Genome:
    weights: Weights = None
    fitness: float = 0.0
    id: int = 0


@dataclass
class SaveState:
    best_for_each_generation: List[Genome]
    genomes: List[Genome]
    current_generation: int
    command_args: dict = None


class GA:
    def __init__(
        self,
        population_size: int,
        generations: int,
        fitness: Callable,
        save_file: str,
        command_args: dict = None,
        genome_workers: int = 1,
    ):
        self.population_size = population_size
        self.generations = generations
        self.fitness = fitness
        self.save_file = save_file
        self.command_args = command_args
        self.genome_workers = genome_workers
        self.best_per_generation = []
        self._pool = None

        self.select_best_n = 15
        self.mutation_rate = 0.05
        self.mutation_step = 0.2
        self.genome_count = 0

    def create_initial(self) -> List[Genome]:
        genomes = []
        for i in range(self.population_size):
            weights = Weights()
            for field in weights.__dict__.keys():
                setattr(weights, field, random.uniform(-1, 1))

            genome = Genome(
                weights=weights,
                fitness=0.0,
                id=self.genome_count,
            )
            genomes.append(genome)
            self.genome_count += 1

        return genomes

    def select_best(self, genomes: List[Genome], progress=True):
        pbar = None
        if progress:
            pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(genomes)).start()

        if self._pool is not None:
            for i, (genome, fitness) in enumerate(
                zip(genomes, self._pool.imap(_evaluate_genome, [g.weights for g in genomes]))
            ):
                genome.fitness = fitness
                if pbar:
                    pbar.update(i + 1)
        else:
            for i, genome in enumerate(genomes):
                genome.fitness = self.fitness(genome.weights)
                if pbar:
                    pbar.update(i + 1)

        if pbar:
            pbar.finish()
        best_performers = sorted(genomes, key=lambda x: x.fitness, reverse=True)
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
                    value = value + random.random() * self.mutation_step * 2 - self.mutation_step
                setattr(child_weights, field, value)
            children.append(Genome(weights=child_weights, id=self.genome_count))
            self.genome_count += 1
        return children

    def run(self, resume: bool = False):
        if resume and os.path.isfile(self.save_file):
            # Save files are written by this same process — trusted local data only.
            with open(self.save_file, "rb") as f:
                save = pickle.load(f)
            genomes = save.genomes
            current = save.current_generation
            self.best_per_generation = save.best_for_each_generation
            saved_args = getattr(save, "command_args", None)
            if saved_args:
                print(f"Originally trained with args: {saved_args}")
        else:
            genomes = self.create_initial()
            current = 0

        if self.genome_workers > 1:
            try:
                multiprocessing.set_start_method("spawn")
            except RuntimeError:
                pass
            print(f"Initializing genome pool with {self.genome_workers} workers")
            self._pool = multiprocessing.Pool(
                self.genome_workers,
                initializer=_genome_worker_init,
                initargs=(self.fitness,),
            )

        try:
            for gen in range(current, self.generations):
                print(f"Generation: {gen}")
                best = self.select_best(genomes)
                genomes = self.combine_and_mutate(best)
                print(best[0])
                self.best_per_generation.append(best[0])
                with open(self.save_file, "wb") as f:
                    pickle.dump(SaveState(self.best_per_generation, genomes, gen + 1, self.command_args), f)
        finally:
            if self._pool is not None:
                self._pool.terminate()
                self._pool.join()
                self._pool = None

        return self.select_best(genomes)[0]
