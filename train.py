#!/usr/bin/env python

import pickle

import gym_tetris
import neat
import numpy as np
from gym_tetris.actions import SIMPLE_MOVEMENT as available_actions
from nes_py.wrappers import JoypadSpace

import board_detect


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


def eval_genome(genome, config, genome_id=None, render=False):
    env = gym_tetris.make("TetrisA-v3")
    env = JoypadSpace(env, available_actions)
    env.reset()

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    done = False
    fitness_current = 0.0
    last_piece = None
    last_actions = []
    actions_to_keep = 30
    new_piece = False
    has_shifted_or_rotated = False
    has_shifted_or_rotated_last_pieces = []

    while not done:
        if render:
            env.render()
        image = env.render("rgb_array")
        cropped_image = image[49:209, 96:176]
        # print(cropped_image.shape)
        # viewer.imshow(cropped_image)
        board = board_detect.detect_board(cropped_image)
        # board.print()
        # print(board.height())

        current_piece = board.detect_current_piece()
        if current_piece:
            new_piece = current_piece != last_piece
            last_piece = current_piece
        if new_piece:
            has_shifted_or_rotated_last_pieces.append(has_shifted_or_rotated)
            has_shifted_or_rotated = False

        piece_to_use = current_piece or last_piece

        board = np.ndarray.flatten(board.board)
        inputs = np.append(board, current_piece_to_id(piece_to_use))
        # print(inputs)
        nnout = net.activate(inputs)
        action = nnout_to_action(nnout)

        print(piece_to_use, available_actions[action])
        if new_piece or available_actions[action] in (["NOOP"], ["down"]):
            last_actions = [action]
        else:
            has_shifted_or_rotated = True
            last_actions.append(action)

        state, rew, done, info = env.step(action)
        fitness_current += rew

        if len(last_actions) > actions_to_keep:
            last_actions.pop(0)
        # print(last_actions)
        # print(has_shifted_or_rotated_last_pieces)
        # if all of the last actions are the same then
        if len(last_actions) >= actions_to_keep and len(set(last_actions)) <= 1:
            fitness_current -= 10
            done = True

        # if last 2 pieces weren't shifted or rotated then end this genome
        if new_piece and len(has_shifted_or_rotated_last_pieces) > 2:
            if not any(has_shifted_or_rotated_last_pieces):
                fitness_current -= 5
                done = True
            has_shifted_or_rotated_last_pieces.pop(0)

        if info["board_height"] > 10 and info["number_of_lines"] == 0:
            fitness_current -= 1
            done = True

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
    p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-315")

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()

    p.add_reporter(stats)

    # Save the process after each 10 frames
    p.add_reporter(neat.Checkpointer(1))

    # pe = neat.ParallelEvaluator(1, eval_genome)
    # winner = p.run(pe.evaluate)

    winner = p.run(eval_genomes)

    with open("winner.pkl", "wb") as output:
        pickle.dump(winner, output, 1)


if __name__ == "__main__":
    main()
