import threading
import time
from collections import Counter

from .board import Board
from .constants import frames_per_cell_by_level
from .pieces import Piece
from .state import GameState

score_by_number_of_lines_cleared = [40, 100, 300, 1200]


class Game:
    def __init__(self, seed=int(time.time() * 1000), level=19, piece_list: [Piece] = None):
        self.state = GameState(seed, piece_list)
        self.state_lock = threading.Lock()
        self.frames_per_cell = frames_per_cell_by_level[level]
        self.state.frames_per_cell = self.frames_per_cell
        self.framerate = 1 / 60
        self.game_over = False
        self.lines = 0
        self.piece_count = 0
        self.game_thread = threading.Thread(target=self.run)
        self.state.board = Board()
        self.score = 0
        self.piece_stats = Counter()
        self.line_combos = Counter()
        self.level = level
        self._starting_level = level
        # NES first-advance threshold: min(L*10+10, max(100, L*10-50))
        # e.g. level 19 → 140 lines before 19→20 (NES counting bug)
        self._first_advance_lines = min(level * 10 + 10, max(100, level * 10 - 50))

    def _advance_level_if_needed(self):
        if self.lines < self._first_advance_lines:
            return
        new_level = self._starting_level + 1 + (self.lines - self._first_advance_lines) // 10
        if new_level != self.level:
            self.level = new_level
            self.frames_per_cell = frames_per_cell_by_level.get(new_level, 1)
            self.state.frames_per_cell = self.frames_per_cell

    def start(self):
        self.game_thread.start()

    def display(self):
        self.state.display()

    def run(self) -> int:
        cp = self.state.select_next_piece()
        self.piece_stats[cp.name] += 1
        self.state_lock.acquire()
        self.state.update(self.state.board, cp)
        self.state_lock.release()
        self.piece_count += 1
        time.sleep(1)
        while not self.game_over:
            frame_start = time.time()
            self.state_lock.acquire()
            moved_down = self.state.move_down()
            self.state.update(self.state.board, cp)
            self.state_lock.release()
            if not moved_down:
                self.game_over = self.state.check_game_over()
                if self.game_over:
                    return self.lines
                else:
                    lines = self.state.check_full_lines()
                    if lines > 0:
                        self.lines += lines
                        self._advance_level_if_needed()
                        if lines <= len(score_by_number_of_lines_cleared):
                            self.score += score_by_number_of_lines_cleared[lines - 1] * (self.level + 1)
                        self.line_combos[lines] += 1
                    cp = self.state.select_next_piece()
                    self.piece_stats[cp.name] += 1
                    self.state_lock.acquire()
                    self.state.update(self.state.board, cp)
                    self.state_lock.release()
                    self.piece_count += 1
            time.sleep(
                max(
                    0,
                    self.framerate * self.frames_per_cell - (time.time() - frame_start),
                )
            )

    def move_down(self, moves: int = 1) -> bool:
        self.state_lock.acquire()
        success = self.state.move_down(moves)
        self.state_lock.release()
        return success

    def move_left(self, moves: int = 1) -> bool:
        self.state_lock.acquire()
        success = self.state.move_left(moves)
        self.state_lock.release()
        return success

    def move_right(self, moves: int = 1) -> bool:
        self.state_lock.acquire()
        success = self.state.move_right(moves)
        self.state_lock.release()
        return success

    def rot_ccw(self, rot: int = 1) -> bool:
        self.state_lock.acquire()
        success = self.state.rot_ccw(rot)
        self.state_lock.release()
        return success

    def rot_cw(self, rot: int = 1) -> bool:
        self.state_lock.acquire()
        success = self.state.rot_cw(rot)
        self.state_lock.release()
        return success

    def move_seq_complete(self):
        self.state_lock.acquire()
        self.state.update(self.state.board, self.state.current_piece)
        self.state_lock.release()
