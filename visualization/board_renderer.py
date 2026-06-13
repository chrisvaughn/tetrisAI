"""Shared rendering of a GameState (board + pieces) to a BGR image.

Used by LiveViewer for live runs (in-memory and emulator) and can be reused
by the replay viewer for recorded runs.
"""

import cv2
import numpy as np

from tetris.pieces import PIECE_COLORS, PIECE_COLORS_BY_ID
from tetris.state import GameState

EMPTY_CELL_COLOR = (20, 20, 20)
FILLED_CELL_COLOR = (100, 100, 100)
GRID_LINE_COLOR = (50, 50, 50)
GHOST_OUTLINE_COLOR = (0, 165, 255)  # Orange outline for planned placement
TEXT_COLOR = (255, 255, 255)


class BoardRenderer:
    """Renders a GameState to a numpy BGR image."""

    def __init__(self, block_size: int = 20, show_info: bool = True, show_next: bool = True):
        self.block_size = block_size
        self.show_info = show_info
        self.show_next = show_next

        self.board_width = 10 * block_size
        self.board_height = 20 * block_size
        self.info_width = 140 if show_info else 0
        self.next_height = 80 if show_next else 0
        self.total_width = self.board_width + self.info_width
        self.total_height = self.board_height + self.next_height

    def render(
        self,
        state: GameState,
        ghost_state: "GameState | None" = None,
        info: "dict | None" = None,
    ) -> np.ndarray:
        img = np.zeros((self.total_height, self.total_width, 3), dtype=np.uint8)

        if state.board is not None:
            self._draw_board(img, state.board.board, state.board.piece_board)
            if ghost_state is not None and ghost_state.board is not None:
                self._draw_ghost(img, state.board.board, ghost_state.board.board)

        if state.current_piece is not None:
            board_arr = state.board.board if state.board is not None else None
            self._draw_piece(
                img, state.current_piece, PIECE_COLORS.get(state.current_piece.name, TEXT_COLOR), board=board_arr
            )

        if self.show_next and state.next_piece is not None:
            self._draw_next_piece(img, state.next_piece)

        if self.show_info and info:
            self._draw_info(img, info)

        return img

    def _draw_board(self, img: np.ndarray, board: np.ndarray, piece_board: np.ndarray = None):
        for y in range(20):
            for x in range(10):
                if board[y, x] == 0:
                    color = EMPTY_CELL_COLOR
                elif piece_board is not None:
                    color = PIECE_COLORS_BY_ID.get(int(piece_board[y, x]), FILLED_CELL_COLOR)
                else:
                    color = FILLED_CELL_COLOR
                x1, y1 = x * self.block_size, y * self.block_size
                x2, y2 = x1 + self.block_size, y1 + self.block_size
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), GRID_LINE_COLOR, 1)

    def _draw_piece(self, img: np.ndarray, piece, color, fill: bool = True, board: np.ndarray = None):
        px, py = piece.zero_based_corner_xy
        for cy, cx in piece.cell_tuples:
            x, y = px + cx, py + cy
            if 0 <= x < 10 and 0 <= y < 20:
                # Skip cells already locked into the board: a transient state
                # can briefly have current_piece pointing at the just-locked
                # piece before the next piece is assigned.
                if board is not None and board[y, x] != 0:
                    continue
                x1, y1 = x * self.block_size, y * self.block_size
                x2, y2 = x1 + self.block_size, y1 + self.block_size
                if fill:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

    def _draw_ghost(self, img: np.ndarray, board_now: np.ndarray, board_ghost: np.ndarray):
        """Outline cells that the bot's plan expects to fill but aren't filled yet."""
        for y in range(20):
            for x in range(10):
                if board_ghost[y, x] != 0 and board_now[y, x] == 0:
                    x1, y1 = x * self.block_size, y * self.block_size
                    x2, y2 = x1 + self.block_size, y1 + self.block_size
                    cv2.rectangle(img, (x1, y1), (x2, y2), GHOST_OUTLINE_COLOR, 2)

    def _draw_next_piece(self, img: np.ndarray, piece):
        shape = piece.shapes[piece.default_shape_idx]
        color = PIECE_COLORS.get(piece.name, TEXT_COLOR)

        where = np.where(shape == 1)
        if len(where[0]) == 0:
            return

        min_y, max_y = np.min(where[0]), np.max(where[0])
        min_x, max_x = np.min(where[1]), np.max(where[1])
        piece_height = max_y - min_y + 1
        piece_width = max_x - min_x + 1

        small_block_size = self.block_size // 2
        y_offset = self.board_height + (self.next_height - piece_height * small_block_size) // 2
        x_offset = (self.board_width - piece_width * small_block_size) // 2

        for dy in range(min_y, max_y + 1):
            for dx in range(min_x, max_x + 1):
                if shape[dy, dx] != 0:
                    x1 = x_offset + (dx - min_x) * small_block_size
                    y1 = y_offset + (dy - min_y) * small_block_size
                    x2, y2 = x1 + small_block_size, y1 + small_block_size
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

    def _draw_info(self, img: np.ndarray, info: dict):
        x_offset = self.board_width + 5
        y = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        for label, value in info.items():
            text = f"{label}: {value}"
            cv2.putText(img, text, (x_offset, y), font, font_scale, TEXT_COLOR, thickness)
            y += 20
