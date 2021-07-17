import copy
import time
from typing import Union

import cv2
import numpy as np

from .board import Board
from .pieces import Piece, Tetrominoes


def nes_prng(value: int):
    bit1 = (value >> 1) & 1
    bit9 = (value >> 9) & 1
    lmb = bit1 ^ bit9
    return (lmb << 15) | (value >> 1)


class GameState:
    def __init__(
        self,
        board: Board,
        current_piece: Union[Piece, None] = None,
        next_piece: Union[Piece, None] = None,
        seed: int = time.time_ns(),
    ):
        self.board = board
        self.current_piece = current_piece
        self.next_piece = next_piece
        self._last_piece: Union[Piece, None] = None
        self._completed_lines: int = 0
        self._last_rn: int = seed
        self._look_for_new_piece: bool = True

    def select_next_piece(self) -> Piece:
        value = nes_prng(self._last_rn)
        self._last_rn = value
        p = Tetrominoes[value % len(Tetrominoes)]
        p.set_position(6, 1)
        return p

    def display(self, in_bounds=False):
        block_size = 28
        x_offset = 5
        y_offset = 5
        virtual_board = np.zeros(
            (
                (Board.rows + (y_offset * 2)) * block_size,
                (Board.columns + (x_offset * 2)) * block_size,
                3,
            ),
            dtype=np.uint8,
        )
        for y, cols in enumerate(self.board.board):
            y = y + y_offset
            for x, cell in enumerate(cols):
                x = x + x_offset
                cv2.rectangle(
                    virtual_board,
                    (x * block_size, y * block_size),
                    ((x + 1) * block_size, (y + 1) * block_size),
                    (255, 255, 255),
                    cv2.FILLED if cell != 0 else None,
                )
        if self.current_piece and in_bounds:
            (sx, sy), (_, _), (w, h) = self.current_piece.in_bounds_rect()
            px, py = self.current_piece.zero_based_corner_xy
            px = px + x_offset
            py = py + y_offset
            for (y, x), value in np.ndenumerate(self.current_piece.shape):
                if sx <= x < sx + w and sy <= y < sy + h:
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
                        (0, 0, 255),
                        cv2.FILLED,
                    )
        if self.current_piece:
            px, py = self.current_piece.zero_based_corner_xy
            px = px + x_offset
            py = py + y_offset
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
                else:
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
                        (0, 255, 0),
                        None,
                    )

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (block_size * 2, block_size * 2 + y_offset)
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
        self._last_piece = self.current_piece
        if current_piece is not None:
            self.current_piece = current_piece
            _, y = self.current_piece.zero_based_corner_xy
            if y > 0:
                self._look_for_new_piece = True
        self.next_piece = next_piece

    def clone(self):
        return copy.deepcopy(self)

    def new_piece(self) -> bool:
        if self._look_for_new_piece and self.current_piece:
            _, y = self.current_piece.zero_based_corner_xy
            if y < 1:
                self._look_for_new_piece = False
                return True
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

    def move_possible(self, p: Piece) -> bool:
        if p.all_segments_in_bounds():
            (sx, sy), (px, py), (w, h) = p.in_bounds_rect()
            return (
                self.board.board[py : py + h, px : px + w]
                * p.shape[sy : sy + h, sx : sx + w]
            ).sum() == 0
        else:
            return False

    def move_down_possible(self, moves: int = 1) -> bool:
        p = self.current_piece.clone()
        p.move_down(moves)
        return self.move_possible(p)

    def move_left_possible(self, moves: int = 1) -> bool:
        p = self.current_piece.clone()
        p.move_left(moves)
        return self.move_possible(p)

    def move_right_possible(self, moves: int = 1) -> bool:
        p = self.current_piece.clone()
        p.move_right(moves)
        return self.move_possible(p)

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
                self.board.board[
                    py + y,
                    px + x,
                ] = value
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

    def count_holes(self) -> int:
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
