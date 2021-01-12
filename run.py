#!/usr/bin/env python

import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT as AVAILABLE_ACTIONS
from nes_py.wrappers import JoypadSpace

from bot import Detectorist, Evaluator
from tetris import GameState


def move_to_action(move):
    if move == "rot_left":
        return AVAILABLE_ACTIONS.index(["A"])
    if move == "rot_right":
        return AVAILABLE_ACTIONS.index(["B"])
    if move == "move_left":
        return AVAILABLE_ACTIONS.index(["left"])
    if move == "move_right":
        return AVAILABLE_ACTIONS.index(["right"])
    if move == "move_down":
        return AVAILABLE_ACTIONS.index(["down"])


def main():
    env = gym_tetris.make("TetrisA-v3")
    env = JoypadSpace(env, AVAILABLE_ACTIONS)
    env.reset()

    gs = None
    move_sequence = []
    done = False
    while not done:
        env.render()

        # print("Running Detectorist")
        image = env.render("rgb_array")
        d = Detectorist(image)
        if not gs:
            print("Building GameState")
            gs = GameState(d.board, d.current_piece, d.next_piece)
        else:
            # print("Updating GameState")
            gs.update(d.board, d.current_piece, d.next_piece)

        if gs.new_piece() and not move_sequence:
            # print("Building out a move sequence")
            aie = Evaluator(gs)
            # move_sequence = aie.random_valid_move_sequence()
            move_sequence = aie.best_move_sequence()
            print(move_sequence)

        if move_sequence:
            action = move_to_action(move_sequence.pop(0))
            # print(availabe_actions[action])
        else:
            action = AVAILABLE_ACTIONS.index(["NOOP"])
        _, _, done, _ = env.step(action)
        if not done:
            state, reward, done, info = env.step(AVAILABLE_ACTIONS.index(["NOOP"]))

    input("Game Over. Press enter to close emulator.")
    env.close()


if __name__ == "__main__":
    main()
