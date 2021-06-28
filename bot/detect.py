import os
from typing import Tuple, Union

import cv2
import numpy as np
import pytesseract
from tetris import Board, Piece, Tetrominoes

cur_dir = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(cur_dir, "templates")
piece_templates = {
    "t": cv2.imread(os.path.join(image_path, "t.png"), cv2.IMREAD_GRAYSCALE),
    "i": cv2.imread(os.path.join(image_path, "i.png"), cv2.IMREAD_GRAYSCALE),
    "z": cv2.imread(os.path.join(image_path, "z.png"), cv2.IMREAD_GRAYSCALE),
    "s": cv2.imread(os.path.join(image_path, "s.png"), cv2.IMREAD_GRAYSCALE),
    "l": cv2.imread(os.path.join(image_path, "l.png"), cv2.IMREAD_GRAYSCALE),
    "j": cv2.imread(os.path.join(image_path, "j.png"), cv2.IMREAD_GRAYSCALE),
    "o": cv2.imread(os.path.join(image_path, "o.png"), cv2.IMREAD_GRAYSCALE),
}


class Detectorist:
    def __init__(self, image: np.ndarray, lines: int, next_piece: int):
        self.image: np.ndarray = image
        self._board: Union[Board, None] = None
        self._current_piece: Union[Piece, None] = None
        self._next_piece: Union[Piece, None] = None
        self._detect_lines_every = lines
        self._detect_next_piece_every = next_piece
        self._detection_count: int = 0
        if np.shape(self.image) != (240, 256):
            raise Exception("image needs to be 256x240 and gray scale")

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

    def update(self, image: np.ndarray):
        self.image = image
        if np.shape(self.image) != (240, 256):
            raise Exception("image needs to be 256x240 and gray scale")

        self._detect()

    def _detect(self):
        self._detect_board()
        self._detect_current_piece()
        if (self._detection_count % self._detect_next_piece_every) == 0:
            self._detect_next_piece()
        # if (self._detection_count % self._detect_lines_every) == 0:
        #     l = self._detect_lines_completed()
        #     print(l)

        self._detection_count += 1

    def _detect_board(self):
        board_image = self.image[47:209, 95:176]
        # cv2.imshow("Board Image", board_image)
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
                    self._current_piece.set_position(
                        at_x - piece.x_offset, at_y + piece.y_offset
                    )
                    return

    def _detect_next_piece(self):
        next_piece_image = self.image[111:144, 191:223]
        for name, template in piece_templates.items():
            res = cv2.matchTemplate(next_piece_image, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.9:
                for piece in Tetrominoes:
                    if piece.name == name:
                        self._next_piece = piece
                        return
        self._next_piece = None

    def _detect_lines_completed(self):
        lines_image = self.image[20:34, 150:178]
        height, width = np.shape(lines_image)
        lines_image = cv2.resize(
            lines_image, dsize=(width * 2, height * 2), interpolation=cv2.INTER_LINEAR
        )
        cv2.imshow("Lines Image", lines_image)
        custom_oem_psm_config = "-l eng --oem 1 --psm 7"
        lines = pytesseract.image_to_string(lines_image, config=custom_oem_psm_config)
        return lines.strip()

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

            # Calculates the mean GBR value from the set of pixels. Uses a set of pixels instead of a
            # single pixel to account for potential issues caused by noise or blurring of a frame.
            mean_gbr = np.array(cv2.mean(block_middle)[0:3])

            # if mean gbr equals black it's the background
            if np.array_equal(mean_gbr, np.array((0, 0, 0))):
                narray[y][x] = 0
            else:
                narray[y][x] = 1

    return narray
