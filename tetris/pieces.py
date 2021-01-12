import copy

import numpy as np


class Piece:
    def __init__(self, name: str, shape: np.ndarray):
        self.name = name
        self.shape = shape
        self.x = 0
        self.y = 0

    def __str__(self) -> str:
        return f"Piece<name: {self.name}, x: {self.x}, y: {self.y}>"

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def move_down(self):
        self.y += 1

    def move_left(self):
        self.x -= 1

    def move_right(self):
        self.x += 1

    def rotate_left(self):
        self.shape = np.rot90(self.shape, 3)

    def rotate_right(self):
        self.shape = np.rot90(self.shape)

    def clone(self):
        return copy.deepcopy(self)


Tetrominoes = [
    Piece("i", np.array([[1, 1, 1, 1]], dtype=int)),
    Piece("j", np.array([[1, 1, 1], [1, 0, 0]], dtype=int)),
    Piece("l", np.array([[1, 1, 1], [0, 0, 1]], dtype=int)),
    Piece("o", np.array([[1, 1], [1, 1]], dtype=int)),
    Piece("s", np.array([[0, 1, 1], [1, 1, 0]], dtype=int)),
    Piece("t", np.array([[1, 1, 1], [0, 1, 0]], dtype=int)),
    Piece("z", np.array([[1, 1, 0], [0, 1, 1]], dtype=int)),
]
