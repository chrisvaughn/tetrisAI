#!/usr/bin/env python

import pickle

import gym_tetris
import neat
import numpy as np
from nes_py.wrappers import JoypadSpace

import board_detect

available_actions = [
    ["right"],
    ["left"],
    ["A"],
    ["down"],
    ["NOOP"],
]


def current_piece_to_id(piece):
    l = [
        None,
        "i",
        "o",
        "t",
        "j",
        "l",
        "s",
        "z",
    ]
    return l.index(piece)


def nnout_to_action(nnout):
    action = nnout.index(max(nnout))
    return action


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, genome_id, True)


def eval_genome(genome, config, genome_id=None, render=False, debug=False):
    env = gym_tetris.make("TetrisA-v3")
    env = JoypadSpace(env, available_actions)
    env.reset()

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    done = False
    fitness_current = 0.0
    last_piece = (None, 0)
    last_action = None
    same_action_count = 0
    last_lines_cleared = 0

    while not done:
        if render:
            env.render()
        image = env.render("rgb_array")
        cropped_image = image[49:209, 96:176]
        board = board_detect.detect_board(cropped_image)

        current_piece = board.detect_current_piece()
        if current_piece[0] is not None and current_piece[0] != last_piece[0]:
            new_piece = True
        else:
            new_piece = False

        if current_piece[0]:
            last_piece = current_piece
            piece_to_use = current_piece
        else:
            piece_to_use = last_piece

        board = np.ndarray.flatten(board.board)
        inputs = np.append(
            board, [current_piece_to_id(piece_to_use[0]), piece_to_use[1]]
        )

        nnout = net.activate(inputs)
        action = nnout_to_action(nnout)
        if debug:
            print(nnout, available_actions[action])
        if last_action == action:
            same_action_count += 1
        else:
            last_action = action
            same_action_count = 0

        # print(piece_to_use, available_actions[action])

        state, rew, done, info = env.step(action)

        # increase by 10 for every line cleared
        fitness_current += (info["number_of_lines"] - last_lines_cleared) * 10
        last_lines_cleared = info["number_of_lines"]

        if new_piece:
            fitness_current += 1

        if info["board_height"] > 15 and info["number_of_lines"] == 0:
            fitness_current -= 1

        if same_action_count > 15 and last_action != available_actions.index(["down"]):
            fitness_current -= 1

        # make this go faster by skipping a few frames
        for i in range(6):
            if not done:
                _, _, done, _ = env.step(available_actions.index(["NOOP"]))

    if genome_id:
        print(f"GenomeID: {genome_id}, Fitness: {fitness_current}")
    else:
        print(f"Fitness: {fitness_current}")

    env.close()
    return fitness_current


def main():
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-feedforward",
    )
    # p = neat.Population(config)
    p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-440")

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()

    p.add_reporter(stats)

    # Save the process after 1 generation
    p.add_reporter(neat.Checkpointer(5))

    pe = neat.ParallelEvaluator(8, eval_genome)
    winner = p.run(pe.evaluate)

    # winner = p.run(eval_genomes)

    with open("winner.pkl", "wb") as output:
        pickle.dump(winner, output, 1)


if __name__ == "__main__":
    main()
