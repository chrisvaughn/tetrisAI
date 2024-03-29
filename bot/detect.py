from typing import Tuple, Union

import numpy as np

from tetris import Board, Piece, Tetrominoes

pixel_threshold_black = 40


class Detectorist:
    def __init__(self):
        self.image: Union[np.ndarray, None] = None
        self._board: Union[Board, None] = None
        self._current_piece: Union[Piece, None] = None
        self._next_piece: Union[Piece, None] = None
        self._detection_count: int = 0

    @property
    def board(self) -> Board:
        return self._board

    @property
    def current_piece(self) -> Piece:
        return self._current_piece

    @property
    def next_piece(self) -> Piece:
        return self._next_piece

    def update(self, image: np.ndarray):
        self.image = image
        if np.shape(self.image) != (240, 256):
            raise Exception("image needs to be 256x240 and gray scale")

        self._detect()

    def _detect(self):
        self._detect_board()
        self._detect_current_piece()
        self._detection_count += 1

    def _detect_board(self):
        if self.image is None:
            raise Exception("image shouldn't be None")
        board_image = self.image[47:209, 95:176]
        # cv2.imshow("Board Image", board_image)
        board = _scan_image(20, 10, board_image)
        self._board = Board(board)

    def _detect_current_piece(self):
        current_piece, at_x, at_y = self._find_current_piece()
        if current_piece is None:
            return
        for piece in Tetrominoes:
            for i, s in enumerate(piece.detection_shapes):
                if np.array_equal(current_piece, s):
                    self._current_piece = piece.clone()
                    self._current_piece.set_detected_position(at_x, at_y, i)
                    return

    def detect_next_piece(self):
        if self.image is None:
            raise Exception("image shouldn't be None")
        next_piece_image = self.image[111:143, 191:223]
        next_piece_arr = _scan_image(4, 4, next_piece_image)
        next_piece_arr, _ = _prune_piece_array(next_piece_arr)
        next_piece = None
        for piece in Tetrominoes:
            if np.array_equal(next_piece_arr, piece.next_piece_detection_shape):
                next_piece = piece.clone()
                break
        self._next_piece = next_piece
        return next_piece

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
    if len(where[0]) == 0:
        return None, 0
    at_x = np.amin(where[1])
    pruned = piece_array[
        np.amin(where[0]) : np.amax(where[0]) + 1,
        np.amin(where[1]) : np.amax(where[1]) + 1,
    ]
    return pruned, at_x


def _scan_image(num_rows: int, num_columns: int, image: np.ndarray) -> np.ndarray:
    narray = np.zeros((num_rows, num_columns), dtype=int)
    height, width = np.shape(image)
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

            # if all pixels are less than the threshold for black it's the background
            if (block_middle < pixel_threshold_black).all():
                narray[y][x] = 0
            else:
                narray[y][x] = 1

    return narray
