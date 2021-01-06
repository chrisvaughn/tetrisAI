"""inspired by https://github.com/Vizaxo/NESTrisRedraw/blob/master/detect/board_block_detect.py"""

import cv2
import numpy as np

tetrominoes = {
    "i": np.array([[1, 1, 1, 1]], dtype=int),
    "o": np.array([[1, 1], [1, 1]], dtype=int),
    "t": np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    "j": np.array([[0, 1], [0, 1], [1, 1]], dtype=int),
    "l": np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    "s": np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
    "z": np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
}


def _prune_piece_array(piece_array):
    first_index = min([y.index(1) for y in piece_array])
    last_index = max([len(y) - 1 - y[::-1].index(1) for y in piece_array])
    pruned = [y[first_index : last_index + 1] for y in piece_array]
    return pruned


def _rotate_piece(piece):
    return list(zip(*piece[::-1]))


class Board:
    def __init__(self, board):
        self.board = board

    def print(self):
        for y in self.board:
            for x in y:
                print("x" if x else " ", end="")
            print("")
        print("\n")

    def height(self):
        h = 0
        for row in reversed(self.board):
            if any(row):
                h += 1
            else:
                break
        return h

    def find_current_piece(self):
        piece_array = []
        for y, row in enumerate(self.board):
            if len(self.board) - y <= self.height():
                break
            if any(row):
                piece_array.append(list(row))

        if not piece_array:
            return None

        piece_array = _prune_piece_array(piece_array)
        return np.array(piece_array, dtype=int)

    def detect_current_piece(self):
        piece = self.find_current_piece()
        if piece is None:
            return None
        for name, shape in tetrominoes.items():
            for i in range(0, 4):
                rp = np.rot90(piece, i)
                if np.array_equal(rp, shape):
                    return name
        return None


def detect_board(image_matrix):
    # Generate a matrix of a blank, standard tetris board (10 x 20 blocks)
    board = np.zeros((20, 10), dtype=int)
    # Calculate information about image matrix
    board_image = image_matrix
    height, width, _ = np.shape(board_image)
    block_height = height / 20
    block_width = width / 10

    for y in range(20):
        for x in range(10):
            x_start_pos = round(x * block_width)
            y_start_pos = round(y * block_height)
            x_end_pos = round(x_start_pos + block_width - 1)
            y_end_pos = round(y_start_pos + block_height - 1)
            # Takes the pixel region containing the current block
            current_block = board_image[
                y_start_pos : y_end_pos + 1, x_start_pos : x_end_pos + 1
            ]

            # Takes a set of 4 (2 x 2) pixels from the center of a block
            block_middle = current_block[
                round(block_height / 2) : round((block_height / 2) + 2),
                round(block_width / 2) : round((block_width / 2) + 2),
            ]

            # Calculates the mean GBR value from the set of pixels. Uses a set of pixels instead of a
            # single pixel to account for potential issues caused by noise or blurring of a frame.
            mean_gbr = np.array(cv2.mean(block_middle)[0:3])

            # if mean gbr equals black it's the background
            if np.array_equal(mean_gbr, np.array((0, 0, 0))):
                board[y][x] = 0
            else:
                board[y][x] = 1

    return Board(board)
