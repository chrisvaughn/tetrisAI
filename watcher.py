#!/usr/bin/env python

import neat

import train

which_genomes = [2634]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        if which_genomes and genome_id in which_genomes:
            genome.fitness = train.eval_genome(genome, config, genome_id, True, True)
        elif not which_genomes:
            genome.fitness = train.eval_genome(genome, config, genome_id, True, True)


def main():
    p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-51")

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.run(eval_genomes)


if __name__ == "__main__":
    main()
