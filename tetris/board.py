import copy
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
        return copy.deepcopy(self)

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

    def _get_peaks(self) -> List[int]:
        if self._peaks:
            return self._peaks

        peaks = [Board.rows] * Board.columns
        for y in range(Board.rows):
            for x in range(Board.columns):
                if self.board[y][x] != 0 and peaks[x] == Board.rows:
                    peaks[x] = y
        self._peaks = peaks
        return peaks

    def count_holes(self) -> (int, int):
        peaks = self._get_peaks()
        holes = 0
        weighted_holes = 0
        for x in range(len(peaks)):
            for y in range(Board.rows - 1, peaks[x], -1):
                if self.board[y][x] == 0:
                    holes += 1
                    weighted_holes += y
        return holes, weighted_holes

    def cumulative_height(self) -> int:
        peaks = self._get_peaks()
        height = 0
        for i in peaks:
            height += 20 - i
        return height

    def roughness(self) -> int:
        peaks = self._get_peaks()
        roughness = 0
        for i in range(len(peaks) - 1):
            roughness += abs(peaks[i] - peaks[i + 1])
        return roughness

    def relative_height(self) -> int:
        peaks = self._get_peaks()
        return max(peaks) - min(peaks)

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
        the number of cells within the wells
        """
        peaks = self._get_peaks()
        well_cells = 0
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
            well_cells += (left - center, right - center)
        return well_cells

    def count_cells(self) -> (int, int):
        peaks = self._get_peaks()
        cells = 0
        weighted_cells = 0
        for x in range(len(peaks)):
            for y in range(Board.rows - 1, peaks[x], -1):
                if self.board[y][x] == 1:
                    cells += 1
                    weighted_cells += y
        return cells, weighted_cells
