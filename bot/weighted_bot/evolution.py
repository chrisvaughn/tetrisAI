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
    piece_lists: list = None
    generation_stats: list = None
    restart_generations: list = None
    last_restart_gen: int = 0


class GA:
    def __init__(
        self,
        population_size: int,
        generations: int,
        fitness: Callable,
        save_file: str,
        command_args: dict = None,
        genome_workers: int = 1,
        piece_lists: list = None,
        seed_weights: List[Weights] = None,
        seeds_per_genome: int = 5,
        seed_noise: float = 0.3,
        restart_noise: float = 2.0,
        restart_random_count: int = 20,
    ):
        self.population_size = population_size
        self.generations = generations
        self.fitness = fitness
        self.save_file = save_file
        self.command_args = command_args
        self.genome_workers = genome_workers
        self.piece_lists = piece_lists
        self._seed_weights = seed_weights or []
        self._seeds_per_genome = seeds_per_genome
        self._seed_noise = seed_noise
        self._restart_noise = restart_noise
        self._restart_random_count = restart_random_count
        self.best_per_generation = []
        self.generation_stats = []
        self.restart_generations = []
        self._pool = None

        self.select_best_n = 20
        self.mutation_rate = 0.15
        self.mutation_step = 0.4
        self.genome_count = 0
        self._last_restart_gen = 0

    def create_initial(self) -> List[Genome]:
        genomes = []

        for seed in self._seed_weights:
            for _ in range(self._seeds_per_genome):
                weights = Weights()
                for field in weights.__dict__.keys():
                    base = getattr(seed, field, 0.0)
                    setattr(weights, field, base + random.gauss(0, self._seed_noise))
                genomes.append(Genome(weights=weights, fitness=0.0, id=self.genome_count))
                self.genome_count += 1

        for _ in range(self.population_size - len(genomes)):
            weights = Weights()
            for field in weights.__dict__.keys():
                setattr(weights, field, random.uniform(-1, 1))
            genomes.append(Genome(weights=weights, fitness=0.0, id=self.genome_count))
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

    def _is_stalled(self, current_gen: int, window: int = 20, min_improvement: float = 0.005) -> bool:
        if current_gen - self._last_restart_gen < window:
            return False
        if len(self.best_per_generation) < window:
            return False
        recent = [g.fitness for g in self.best_per_generation[-window:]]
        oldest, newest = recent[0], recent[-1]
        if oldest == 0:
            return newest == 0
        return (newest - oldest) / abs(oldest) < min_improvement

    def _restart_from_best(self, best_genomes: List[Genome]) -> List[Genome]:
        # Keep all elites unchanged.
        children = list(best_genomes)
        random_count = min(self._restart_random_count, self.population_size - len(best_genomes))
        variant_slots = self.population_size - len(best_genomes) - random_count
        # Distribute variant slots evenly across all seeds so each elite contributes.
        per_seed = variant_slots // len(best_genomes)
        remainder = variant_slots % len(best_genomes)
        for i, seed in enumerate(best_genomes):
            count = per_seed + (1 if i < remainder else 0)
            for _ in range(count):
                child_weights = Weights()
                for field in child_weights.__dict__.keys():
                    value = getattr(seed.weights, field) + random.gauss(0, self._restart_noise)
                    setattr(child_weights, field, value)
                children.append(Genome(weights=child_weights, id=self.genome_count))
                self.genome_count += 1
        # Fresh random genomes to force exploration outside the current plateau.
        for _ in range(random_count):
            child_weights = Weights()
            for field in child_weights.__dict__.keys():
                setattr(child_weights, field, random.uniform(-1, 1))
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
            saved_piece_lists = getattr(save, "piece_lists", None)
            if saved_piece_lists:
                self.piece_lists = saved_piece_lists
                print(f"Restored {len(self.piece_lists)} piece lists from save file")
            self.generation_stats = getattr(save, "generation_stats", None) or []
            self.restart_generations = getattr(save, "restart_generations", None) or []
            self._last_restart_gen = getattr(save, "last_restart_gen", 0)
        else:
            genomes = self.create_initial()
            current = 1
            # Write initial snapshot before evaluation so visualizer has something to show.
            with open(self.save_file, "wb") as f:
                pickle.dump(
                    SaveState(
                        self.best_per_generation,
                        genomes,
                        current,
                        self.command_args,
                        self.piece_lists,
                        self.generation_stats,
                        self.restart_generations,
                        self._last_restart_gen,
                    ),
                    f,
                )

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
            for gen in range(current, self.generations + 1):
                print(f"Generation: {gen}")
                best = self.select_best(genomes)
                fitnesses = [g.fitness for g in genomes]
                mean = sum(fitnesses) / len(fitnesses)
                std = (sum((f - mean) ** 2 for f in fitnesses) / len(fitnesses)) ** 0.5
                stats = {"gen": gen, "best": best[0].fitness, "mean": mean, "std": std}
                self.generation_stats.append(stats)
                print(f"  best={stats['best']:.2f}  mean={mean:.2f}  std={std:.2f}")
                print(best[0])
                self.best_per_generation.append(best[0])
                if self._is_stalled(gen):
                    print(
                        f"Stall detected at generation {gen}, restarting from top {len(best)} genomes "
                        f"(noise={self._restart_noise}, random={self._restart_random_count})"
                    )
                    self.restart_generations.append(gen)
                    self._last_restart_gen = gen
                    genomes = self._restart_from_best(best)
                else:
                    genomes = self.combine_and_mutate(best)
                with open(self.save_file, "wb") as f:
                    pickle.dump(
                        SaveState(
                            self.best_per_generation,
                            genomes,
                            gen + 1,
                            self.command_args,
                            self.piece_lists,
                            self.generation_stats,
                            self.restart_generations,
                            self._last_restart_gen,
                        ),
                        f,
                    )
        finally:
            if self._pool is not None:
                self._pool.terminate()
                self._pool.join()
                self._pool = None

        return self.select_best(genomes)[0]
