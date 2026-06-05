from typing import List

import numpy as np


class Board:
    columns: int = 10
    rows: int = 20

    def __init__(self, board: np.ndarray = None):
        if board is None:
            board = np.zeros((Board.rows, Board.columns), dtype=int)
        self.board = board
        self._peaks = None

    def clone(self):
        new_board = Board.__new__(Board)
        new_board.board = self.board.copy()
        new_board._peaks = None
        return new_board

    def compare(self, other: "Board") -> bool:
        return np.array_equal(self.board, other.board)

    def print(self):
        print("-" * 12)
        for y in self.board:
            print("|", end="")
            for x in y:
                print(f"{x}", end="")
            print("|")
        print("-" * 12)

    def updated(self):
        self._peaks = None

    def height(self) -> int:
        h = 0
        for row in reversed(self.board):
            if any(row):
                h += 1
            else:
                break
        return h

    def _get_peaks(self) -> np.ndarray:
        if self._peaks is not None:
            return self._peaks

        nonzero = self.board != 0
        has_content = nonzero.any(axis=0)
        peaks = np.where(has_content, nonzero.argmax(axis=0), Board.rows)
        self._peaks = peaks
        return peaks

    def count_holes(self) -> (int, int):
        peaks = self._get_peaks()
        rows = np.arange(Board.rows)[:, np.newaxis]
        below_peak = rows > peaks
        hole_mask = below_peak & (self.board == 0)
        holes = int(hole_mask.sum())
        weighted_holes = int((rows * hole_mask).sum())
        return holes, weighted_holes

    def cumulative_height(self) -> int:
        return int(np.sum(Board.rows - self._get_peaks()))

    def roughness(self) -> int:
        return int(np.sum(np.abs(np.diff(self._get_peaks()))))

    def relative_height(self) -> int:
        peaks = self._get_peaks()
        return int(peaks.max() - peaks.min())

    def game_over(self) -> bool:
        return self.board.all()

    def deep_well_count(self) -> int:
        """
        the number of wells containing 3 or more well cells.
        """
        peaks = self._get_peaks()
        well_count = 0
        for i in range(len(peaks) - 1):
            if i == 0:
                left = 0
            else:
                left = peaks[i - 1]

            center = peaks[i]

            if i < 9:
                right = peaks[i + 1]
            else:
                right = 0
            if left - center <= -3 and right - center <= -3:
                well_count += 1
        return well_count

    def count_well_cells(self) -> int:
        """
        Total depth of all wells: for each column deeper than both neighbors,
        the contribution is min(center - left, center - right) in row units.
        Walls are treated as peak row 0 (full height).
        """
        peaks = self._get_peaks()
        padded = np.pad(peaks, 1, constant_values=0)
        depth = np.minimum(peaks - padded[:-2], peaks - padded[2:])
        return int(depth[depth > 0].sum())

    def count_cells(self) -> (int, int):
        peaks = self._get_peaks()
        rows = np.arange(Board.rows)[:, np.newaxis]
        below_peak = rows > peaks
        cell_mask = below_peak & (self.board == 1)
        cells = int(cell_mask.sum())
        weighted_cells = int((rows * cell_mask).sum())
        return cells, weighted_cells

    def spawn_zone_filled(self) -> int:
        # Rows 0-1, columns 3-6 (0-indexed) cover the piece spawn area (center x=6 1-based).
        return int(self.board[0:2, 3:7].sum())

    def count_row_transitions(self) -> int:
        padded = np.pad(self.board, ((0, 0), (1, 1)), constant_values=1)
        return int((np.diff(padded, axis=1) != 0).sum())

    def count_unreachable_cells(self) -> int:
        """Count empty cells sealed off from the top by filled cells."""
        board = self.board
        rows, cols = board.shape
        visited = np.zeros((rows, cols), dtype=bool)
        stack = []

        for c in range(cols):
            if board[0, c] == 0:
                visited[0, c] = True
                stack.append((0, c))

        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and board[nr, nc] == 0:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

        empty = board == 0
        return int((empty & ~visited).sum())
