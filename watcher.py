#!/usr/bin/env python

import neat

import train


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = train.eval_genome(genome, config, genome_id, True, True)


def main():
    p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-903")

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.run(eval_genomes)


if __name__ == "__main__":
    main()
