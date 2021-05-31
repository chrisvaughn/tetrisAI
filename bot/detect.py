from typing import Tuple, Union

import cv2
import numpy as np
from tetris import Board, Piece, Tetrominoes


class Detectorist:
    def __init__(self, image: np.ndarray):
        self.image: np.ndarray = image
        self._board: Union[Board, None] = None
        self._current_piece: Union[Piece, None] = None
        self._next_piece: Union[Piece, None] = None

        self._detect()

    @property
    def board(self) -> Board:
        return self._board

    @property
    def current_piece(self) -> Piece:
        return self._current_piece

    @property
    def next_piece(self) -> Piece:
        return self._next_piece

    def _detect(self):
        self._detect_board()
        self._detect_current_piece()
        self._detect_next_piece()

    def _detect_board(self):
        h = self.image.shape[0]
        w = self.image.shape[1]
        board_width_start = w / 2.6666
        board_width_end = board_width_start + w / 3.2
        board_height_start = h / 6
        board_height_end = h - (h / 9.4)
        board_image = self.image[
            int(board_height_start) : int(board_height_end),
            int(board_width_start) : int(board_width_end),
        ]
        board = _scan_image(20, 10, board_image)
        self._board = Board(board)

    def _detect_current_piece(self):
        current_piece, at_x, at_y = self._find_current_piece()
        if current_piece is None:
            return
        for piece in Tetrominoes:
            for i in range(0, 4):
                rp = np.rot90(current_piece, i)
                if np.array_equal(rp, piece.detection_shape):
                    self._current_piece = piece.clone()
                    self._current_piece.set_position(at_x, at_y)
                    return

    def _detect_next_piece(self):
        h = self.image.shape[0]
        w = self.image.shape[1]
        width_start = 2 * (w / 2.6666)
        width_end = width_start + ((w / 2.6666) / 3)
        height_start = h / 2.1
        height_end = (h / 2.1) + h / 8
        next_piece_image = self.image[
            int(height_start) : int(height_end),
            int(width_start) : int(width_end),
        ]
        next_image_arr = _scan_image(4, 4, next_piece_image)
        if not next_image_arr.any():
            self._next_piece = None
            return
        pruned, _ = _prune_piece_array(next_image_arr)
        for piece in Tetrominoes:
            if np.array_equal(pruned, piece.detection_shape):
                self._next_piece = piece
                return

    def _find_current_piece(self) -> Tuple[Union[np.ndarray, None], int, int]:
        piece_array = np.empty((0, Board.columns), int)
        at_y = -1
        for y, row in enumerate(self._board.board):
            if len(self._board.board) - y <= self._board.height():
                break
            if any(row):
                if at_y < 0:
                    at_y = y
                piece_array = np.append(piece_array, np.array([row]), axis=0)
                # erase row with current piece
                self._board.board[y] = np.zeros(Board.columns, dtype=int)

        if piece_array.size == 0:
            return None, 0, 0

        piece_array, at_x = _prune_piece_array(piece_array)
        return piece_array, at_x, at_y


def _prune_piece_array(piece_array: np.ndarray) -> Tuple[np.ndarray, int]:
    where = np.where(piece_array == 1)
    at_x = np.amin(where[1])
    pruned = piece_array[
        np.amin(where[0]) : np.amax(where[0]) + 1,
        np.amin(where[1]) : np.amax(where[1]) + 1,
    ]
    return pruned, at_x


def _scan_image(num_rows: int, num_columns: int, image: np.ndarray) -> np.ndarray:
    narray = np.zeros((num_rows, num_columns), dtype=int)
    height, width, _ = np.shape(image)
    block_height = height / num_rows
    block_width = width / num_columns
    for y in range(num_rows):
        for x in range(num_columns):
            x_start_pos = round(x * block_width)
            y_start_pos = round(y * block_height)
            x_end_pos = round(x_start_pos + block_width - 1)
            y_end_pos = round(y_start_pos + block_height - 1)
            # Takes the pixel region containing the current block
            current_block = image[
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
                narray[y][x] = 0
            else:
                narray[y][x] = 1

    return narray
