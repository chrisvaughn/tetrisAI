#!/usr/bin/env python

import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT as AVAILABLE_ACTIONS
from nes_py.wrappers import JoypadSpace

from bot import Detectorist, Evaluator, execute_move
from tetris import GameState


def move_to_action(move):
    if move == "rot_ccw":
        return AVAILABLE_ACTIONS.index(["B"])
    if move == "move_left":
        return AVAILABLE_ACTIONS.index(["left"])
    if move == "move_right":
        return AVAILABLE_ACTIONS.index(["right"])
    if move == "move_down":
        return AVAILABLE_ACTIONS.index(["down"])
    if move == "noop":
        return AVAILABLE_ACTIONS.index(["NOOP"])


def main():
    env = gym_tetris.make("TetrisA-v3")
    env = JoypadSpace(env, AVAILABLE_ACTIONS)
    env.reset()

    gs = None
    move_sequence = []
    done = False
    while not done:
        env.render()

        image = env.render("rgb_array")
        d = Detectorist(image)
        if not gs:
            print("Building GameState")
            gs = GameState(d.board, d.current_piece, d.next_piece)
        else:
            # print("Updating GameState")
            gs.update(d.board, d.current_piece, d.next_piece)

        if gs.new_piece() and not move_sequence:
            weights = {
                "holes": -0.4,
                "roughness": -0.2,
                "lines": 0.8,
                "height": -0.5,
            }
            aie = Evaluator(gs, weights)
            # move_sequence = aie.random_valid_move_sequence()
            best_move, time_taken = aie.best_move()
            print(f"Move Found in {time_taken} sec")
            print(f"Best Move: {best_move}")
            # temp_state = gs.clone()
            # execute_move(temp_state, best_move.rotations, best_move.translation)
            # temp_state.board.print()
            move_sequence = best_move.to_sequence()
            print(move_sequence)
            # input("Press enter to execute move.")

        if move_sequence:
            action = move_to_action(move_sequence.pop(0))
        else:
            action = AVAILABLE_ACTIONS.index(["down"])

        _, _, done, _ = env.step(action)
        if not done:
            state, reward, done, info = env.step(AVAILABLE_ACTIONS.index(["down"]))

    input("Game Over. Press enter to close emulator.")
    env.close()


if __name__ == "__main__":
    main()
