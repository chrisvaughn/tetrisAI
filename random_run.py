#!/usr/bin/env python

import cv2
import gym_tetris
import numpy as np
from gym.envs.classic_control import rendering
from gym_tetris.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import board_detect

env = gym_tetris.make("TetrisA-v3")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
viewer = rendering.SimpleImageViewer()

state = env.reset()

done = False
last_piece = None
for step in range(5000):
    if done:
        state = env.reset()
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
        last_piece = current_piece

    print(current_piece or last_piece)
    action = env.action_space.sample()
    # print(action)
    state, reward, done, info = env.step(action)
    # env.render()

env.close()
