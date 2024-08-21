import copy
from typing import Union

import cv2
import numpy as np

from .board import Board
from .pieces import Piece, Tetrominoes


class InvalidMove(Exception):
    def __init__(self, piece: Piece):
        self.piece = piece

    def __str__(self):
        return f"Invalid move: {self.piece}"


def nes_prng(value: int):
    bit1 = (value >> 1) & 1
    bit9 = (value >> 9) & 1
    lmb = bit1 ^ bit9
    return (lmb << 15) | (value >> 1)


class GameState:
    def __init__(self, seed: int = 0, piece_list=None):
        self.board: Union[Board, None] = None
        self.current_piece: Union[Piece, None] = None
        self.next_piece: Union[Piece, None] = None
        self._last_piece_y: int = 0
        self._completed_lines: int = 0
        self._last_rn: int = seed
        self._first_piece: bool = True
        self.piece_list = piece_list

    def select_next_piece(self) -> Piece:
        if self.piece_list:
            return self.piece_list.pop(0)

        value = nes_prng(self._last_rn)
        self._last_rn = value
        p = Tetrominoes[value % len(Tetrominoes)].clone()
        p.set_position(6, 1)
        return p

    def display(self):
        block_size = 28
        virtual_board = np.zeros(
            (Board.rows * block_size, Board.columns * block_size, 3), dtype=np.uint8
        )
        for y, cols in enumerate(self.board.board):
            for x, cell in enumerate(cols):
                cv2.rectangle(
                    virtual_board,
                    (x * block_size, y * block_size),
                    ((x + 1) * block_size, (y + 1) * block_size),
                    (255, 255, 255),
                    cv2.FILLED if cell != 0 else None,
                )
        if self.current_piece:
            px, py = self.current_piece.zero_based_corner_xy
            for (y, x), value in np.ndenumerate(self.current_piece.shape):
                if value != 0:
                    cv2.rectangle(
                        virtual_board,
                        (
                            (px + x) * block_size,
                            (py + y) * block_size,
                        ),
                        (
                            (px + x + 1) * block_size,
                            (py + y + 1) * block_size,
                        ),
                        (255, 0, 0),
                        cv2.FILLED,
                    )

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (block_size * 2, block_size * 2)
        font_scale = 1
        color = (255, 255, 0)
        thickness = 2

        cv2.putText(
            virtual_board,
            str(self._completed_lines),
            org,
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        cv2.imshow("Virtual Board", virtual_board)

    def update(self, board: Board, current_piece: Piece, next_piece: Piece = None):
        self.board = board
        if self.current_piece:
            self._last_piece_y = self.current_piece.zero_based_corner_xy[1]
        if current_piece:
            self.current_piece = current_piece
        self.next_piece = next_piece

    def clone(self):
        return copy.deepcopy(self)

    def new_piece(self) -> bool:
        if self.current_piece and self._first_piece:
            self._first_piece = False
            return True
        if self.current_piece:
            return self.current_piece.zero_based_corner_xy[1] < self._last_piece_y
        return False

    def move_down(self, moves: int = 1):
        if self.move_down_possible(moves):
            self.current_piece.move_down(moves)
            return True
        else:
            if moves == 1:
                self.place_piece_on_board()
            return False

    def move_left(self, moves: int = 1):
        if self.move_left_possible(moves):
            self.current_piece.move_left(moves)
            return True
        return False

    def move_right(self, moves: int = 1) -> bool:
        if self.move_right_possible(moves):
            self.current_piece.move_right(moves)
            return True
        return False

    def move_down_possible(self, moves: int = 1) -> bool:
        px, py = self.current_piece.zero_based_corner_xy
        py = py + moves
        for (y, x), value in np.ndenumerate(self.current_piece.shape):
            if value != 0 and py + y >= 0:
                if (
                    py + y >= self.board.rows
                    or px + x >= self.board.columns
                    or self.board.board[py + y, px + x] != 0
                ):
                    return False
        return True

    def move_left_possible(self, moves: int = 1) -> bool:
        px, py = self.current_piece.zero_based_corner_xy
        px = px - moves
        for (y, x), value in np.ndenumerate(self.current_piece.shape):
            if value != 0 and py + y >= 0:
                if px + x < 0 or self.board.board[py + y, px + x] != 0:
                    return False
        return True

    def move_right_possible(self, moves: int = 1) -> bool:
        px, py = self.current_piece.zero_based_corner_xy
        px = px + moves
        for (y, x), value in np.ndenumerate(self.current_piece.shape):
            if value != 0 and px + x >= 0 and py + y >= 0:
                if (
                    px + x >= self.board.columns
                    or self.board.board[py + y, px + x] != 0
                ):
                    return False
        return True

    def rot_ccw_possible(self, rot: int = 1):
        p = self.current_piece.clone()
        p.rot_ccw(rot)
        return self.rot_possible(p.shape)

    def rot_cw_possible(self, rot: int = 1):
        p = self.current_piece.clone()
        p.rot_cw(rot)
        return self.rot_possible(p.shape)

    def rot_possible(self, shape) -> bool:
        px, py = self.current_piece.zero_based_corner_xy
        for (y, x), value in np.ndenumerate(shape):
            if value != 0 and py + y >= 0:
                if (
                    py + y >= self.board.rows
                    or px + x < 0
                    or px + x >= self.board.columns
                    or self.board.board[py + y, px + x] != 0
                ):
                    return False
        return True

    def rot_ccw(self, rot: int = 1) -> bool:
        if self.rot_ccw_possible(rot):
            self.current_piece.rot_ccw(rot)
            return True
        return False

    def rot_cw(self, rot: int = 1):
        if self.rot_cw_possible(rot):
            self.current_piece.rot_cw(rot)
            return True
        return False

    def place_piece_on_board(self):
        px, py = self.current_piece.zero_based_corner_xy
        for (y, x), value in np.ndenumerate(self.current_piece.shape):
            if value != 0:
                try:
                    self.board.board[
                        py + y,
                        px + x,
                    ] = value
                except IndexError:
                    raise InvalidMove(self.current_piece)

        self.board.updated()

    def check_game_over(self):
        return np.any(self.board.board[0] != 0)

    def check_full_lines(self) -> int:
        full_lines = np.count_nonzero(np.all(self.board.board != 0, 1))
        if full_lines != 0:
            removed_lines = self.board.board[np.all(self.board.board != 0, 1)]
            removed_lines.fill(0)
            clean_board = self.board.board[np.any(self.board.board == 0, 1)]
            self.board.board = np.vstack((removed_lines, clean_board))
        self._completed_lines += full_lines
        return full_lines

    def count_holes(self) -> (int, int):
        return self.board.count_holes()

    def cumulative_height(self) -> int:
        return self.board.cumulative_height()

    def roughness(self) -> int:
        return self.board.roughness()

    def relative_height(self) -> int:
        return self.board.relative_height()

    def absolute_height(self) -> int:
        return self.board.height()

    def well_count(self) -> int:
        return self.board.well_count()
