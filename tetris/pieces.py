import copy
from typing import List, Tuple

import numpy as np

from .board import Board


class Piece:
    def __init__(self, name: str, shapes: List[np.ndarray], default_shape_idx: int):
        self.name = name
        self.shapes = shapes
        self.default_shape_idx = default_shape_idx
        self._x = 0  # the center of the shape matrix
        self._y = 0  # the center of the shape matrix
        self.current_shape_idx = default_shape_idx
        self._detection_shapes = None

    def __str__(self) -> str:
        return f"Piece<name: {self.name}, x: {self._x}, y: {self._y}>"

    @property
    def shape(self) -> np.ndarray:
        return self.shapes[self.current_shape_idx]

    @property
    def corner_xy(self) -> Tuple[int, int]:
        return self._x - 2, self._y - 2

    @property
    def zero_based_corner_xy(self) -> Tuple[int, int]:
        return self._x - 2 - 1, self._y - 2 - 1

    # 0-based, left most x, top most y, number of rotations of detection shape
    def set_detected_position(self, x: int, y: int, shape_idx: int):
        self.current_shape_idx = shape_idx
        where = np.where(self.shape == 1)
        self._x = x + 2 - np.amin(where[1]) + 1
        self._y = y + 2 - np.amin(where[0]) + 1

    # 1-based, center of piece
    def set_position(self, x: int, y: int):
        if x < 1 or x > Board.columns:
            raise ValueError(f"x must be between 1 and {Board.columns}")
        if y < 1 or y > Board.rows:
            raise ValueError(f"y must be between 1 and {Board.rows}")
        self._x = x
        self._y = y

    def possible_translations(self) -> Tuple[int, int]:
        where = np.where(self.shape == 1)
        min_x = np.amin(where[1])
        max_x = np.amax(where[1])
        t_left = self._x - (self.shape.shape[1] // 2 - min_x) - 1
        t_right = 10 - self._x + (self.shape.shape[1] // 2 - max_x)
        return t_left, t_right

    def move_down(self, moves: int = 1):
        self._y += moves

    def move_left(self, moves: int = 1):
        self._x -= moves

    def move_right(self, moves: int = 1):
        self._x += moves

    def rot_ccw(self, rot: int = 1):
        self.current_shape_idx = (self.current_shape_idx - rot) % len(self.shapes)

    def rot_cw(self, rot: int = 1):
        self.current_shape_idx = (self.current_shape_idx + rot) % len(self.shapes)

    def clone(self):
        return copy.deepcopy(self)

    @property
    def detection_shapes(self) -> List[np.ndarray]:
        if self._detection_shapes:
            return self._detection_shapes

        ds = []
        for s in self.shapes:
            where = np.where(s == 1)
            pruned = s[
                np.amin(where[0]) : np.amax(where[0]) + 1,
                np.amin(where[1]) : np.amax(where[1]) + 1,
            ]
            ds.append(pruned)
        self._detection_shapes = ds
        return self._detection_shapes


Tetrominoes = [
    Piece(
        "i",
        [
            np.array(
                [
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
        ],
        1,
    ),
    Piece(
        "l",
        [
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
        ],
        1,
    ),
    Piece(
        "j",
        [
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
        ],
        3,
    ),
    Piece(
        "o",
        [
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            )
        ],
        0,
    ),
    Piece(
        "s",
        [
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
        ],
        0,
    ),
    Piece(
        "t",
        [
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
        ],
        2,
    ),
    Piece(
        "z",
        [
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
        ],
        0,
    ),
]
