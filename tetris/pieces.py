import copy
from typing import Union

import numpy as np


class Piece:
    def __init__(
        self,
        name: str,
        shape: np.ndarray,
        valid_rots: int,
        detection_shape: Union[np.ndarray, None] = None,
    ):
        self.name = name
        self.shape = shape
        self.valid_rotations = valid_rots
        self.detection_shape = detection_shape
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

    def rotate_ccw(self):
        self.shape = np.rot90(self.shape)

    def clone(self):
        return copy.deepcopy(self)


Tetrominoes = [
    Piece(
        "i",
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=int),
        1,
        np.array([[1, 1, 1, 1]], dtype=int),
    ),
    Piece(
        "j",
        np.array([[0, 0, 0], [2, 2, 2], [2, 0, 0]], dtype=int),
        3,
        np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    ),
    Piece(
        "l",
        np.array([[0, 0, 0], [3, 3, 3], [0, 0, 3]], dtype=int),
        3,
        np.array([[1, 1, 1], [0, 0, 1]], dtype=int),
    ),
    Piece(
        "o",
        np.array([[4, 4], [4, 4]], dtype=int),
        0,
        np.array([[1, 1], [1, 1]], dtype=int),
    ),
    Piece(
        "s",
        np.array([[0, 0, 0], [0, 5, 5], [5, 5, 0]], dtype=int),
        3,
        np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
    ),
    Piece(
        "t",
        np.array([[0, 0, 0], [6, 6, 6], [0, 6, 0]], dtype=int),
        3,
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    ),
    Piece(
        "z",
        np.array([[0, 0, 0], [7, 7, 0], [0, 7, 7]], dtype=int),
        3,
        np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    ),
]
