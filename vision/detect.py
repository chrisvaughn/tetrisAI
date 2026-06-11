from typing import Tuple, Union

import numpy as np

from tetris import Board, Piece, Tetrominoes

pixel_threshold_black = 40

# Consecutive frames a piece must be stationary before we treat it as locked.
# This is a secondary debounce on top of the resting-position check in
# _piece_resting() below, which is what actually prevents a piece that's
# still falling (but momentarily appears stationary due to capture timing,
# e.g. while Python is blocked sending a multi-keypress sequence) from being
# baked into the locked board prematurely.
_LOCK_CONFIRMATION_FRAMES = 2

# (y_start, y_end, x_start, x_end) pixel regions within the 256x240 capture
NESTOPIA_OFFSETS = {
    "board": (47, 209, 95, 176),
    "next_piece": (111, 143, 191, 223),
}
FCEUX_OFFSETS = {
    # 160×80 px gives exactly 8×8 px per NES tile — no fractional rounding that
    # causes the bottom 4-5 rows to produce false-positive detections when an
    # adjacent tile's bright top-border pixel lands inside a cell's inner region.
    "board": (55, 215, 95, 175),
    "next_piece": (119, 151, 191, 223),
}


class Detectorist:
    def __init__(self, offsets: dict = None):
        if offsets is None:
            offsets = NESTOPIA_OFFSETS
        self._board_region: Tuple[int, int, int, int] = offsets["board"]
        self._next_piece_region: Tuple[int, int, int, int] = offsets["next_piece"]
        self.image: Union[np.ndarray, None] = None
        self._board: Union[Board, None] = None
        self._current_piece: Union[Piece, None] = None
        self._next_piece: Union[Piece, None] = None
        self._detection_count: int = 0
        # Temporal board/piece separation (option-2 tracker)
        self._locked_board_arr = np.zeros((Board.rows, Board.columns), dtype=int)
        self._prev_piece_candidates: Union[np.ndarray, None] = None
        self._no_movement_frames: int = 0

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
        by0, by1, bx0, bx1 = self._board_region
        board_image = self.image[by0:by1, bx0:bx1]
        board = _scan_image(20, 10, board_image, use_max=True)
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
        ny0, ny1, nx0, nx1 = self._next_piece_region
        next_piece_image = self.image[ny0:ny1, nx0:nx1]
        next_piece_arr = _scan_image(4, 4, next_piece_image)
        next_piece_arr, _ = _prune_piece_array(next_piece_arr)
        next_piece = None
        for piece in Tetrominoes:
            if np.array_equal(next_piece_arr, piece.next_piece_detection_shape):
                next_piece = piece.clone()
                break
        self._next_piece = next_piece
        return next_piece

    def _bootstrap_locked_board(self, board_arr: np.ndarray) -> np.ndarray:
        """Seed _locked_board_arr on the first frame using contiguous-bottom-height."""
        h = 0
        for row in reversed(board_arr):
            if any(row):
                h += 1
            else:
                break
        locked = np.zeros_like(board_arr)
        if h > 0:
            locked[Board.rows - h :] = board_arr[Board.rows - h :]
        return locked

    def _piece_resting(self, piece_candidates: np.ndarray) -> bool:
        """True if the candidate cells' lowest row sits on the floor or directly
        on top of the known-locked stack, i.e. the piece could plausibly have
        landed. A piece still falling high above the stack can never satisfy
        this, regardless of how many identical frames were captured."""
        piece_rows = np.where(piece_candidates.any(axis=1))[0]
        piece_bottom = piece_rows.max()
        if piece_bottom == Board.rows - 1:
            return True
        locked_rows = np.where(self._locked_board_arr.any(axis=1))[0]
        if locked_rows.size == 0:
            return False
        locked_top = locked_rows.min()
        return piece_bottom + 1 >= locked_top

    def _split_piece_groups(self, piece_candidates: np.ndarray) -> list:
        """Split candidate cells into row-contiguous groups (top to bottom),
        returned as (start_row, end_row_exclusive) tuples."""
        groups = []
        in_group = False
        start = 0
        for y in range(Board.rows):
            has = np.any(piece_candidates[y])
            if has and not in_group:
                start = y
                in_group = True
            elif not has and in_group:
                groups.append((start, y))
                in_group = False
        if in_group:
            groups.append((start, Board.rows))
        return groups

    def _find_current_piece(self) -> Tuple[Union[np.ndarray, None], int, int]:
        curr_raw = self._board.board.copy()

        # On the first frame _locked_board_arr is all zeros; seed it spatially so
        # we have a sensible baseline before temporal tracking kicks in.
        if not np.any(self._locked_board_arr) and np.any(curr_raw):
            self._locked_board_arr = self._bootstrap_locked_board(curr_raw)

        # Cells not yet in the known-locked region are candidates for the current piece.
        piece_candidates = curr_raw & ~self._locked_board_arr

        # Nothing above the locked board → ARE / idle period; refresh locked state.
        if not np.any(piece_candidates):
            self._locked_board_arr = curr_raw.copy()
            self._prev_piece_candidates = None
            self._no_movement_frames = 0
            return None, 0, 0

        # If the candidates split into multiple row-separated groups, the
        # bottom-most group may be the previous piece that has already landed
        # but hasn't been folded into _locked_board_arr yet — e.g. the new
        # piece spawned during the lock-confirmation window of the previous
        # piece. If that bottom group is resting on the stack/floor, lock it
        # immediately so the remaining (upper) group is recognized as the new
        # piece at/near its true spawn position, instead of only being
        # detected once it has fallen further.
        groups = self._split_piece_groups(piece_candidates)
        if len(groups) > 1:
            bottom_start, bottom_end = groups[-1]
            bottom_mask = np.zeros_like(piece_candidates)
            bottom_mask[bottom_start:bottom_end] = piece_candidates[bottom_start:bottom_end]
            if self._piece_resting(bottom_mask):
                self._locked_board_arr[bottom_start:bottom_end] |= piece_candidates[bottom_start:bottom_end]
                piece_candidates = piece_candidates.copy()
                piece_candidates[bottom_start:bottom_end] = 0
                self._prev_piece_candidates = None
                self._no_movement_frames = 0
                if not np.any(piece_candidates):
                    return None, 0, 0

        # Count consecutive frames where the candidate mask hasn't changed.
        if self._prev_piece_candidates is not None and np.array_equal(piece_candidates, self._prev_piece_candidates):
            self._no_movement_frames += 1
            if self._no_movement_frames >= _LOCK_CONFIRMATION_FRAMES and self._piece_resting(piece_candidates):
                # Piece has been stationary long enough and is resting on the
                # stack/floor — treat it as locked.
                self._locked_board_arr = curr_raw.copy()
                self._prev_piece_candidates = None
                self._no_movement_frames = 0
                return None, 0, 0
        else:
            self._no_movement_frames = 0

        self._prev_piece_candidates = piece_candidates.copy()

        # Remove piece candidates from the board so it only shows locked cells.
        for y in range(Board.rows):
            if np.any(piece_candidates[y]):
                self._board.board[y] = self._board.board[y] & self._locked_board_arr[y]

        # Build the piece array from candidate rows.
        piece_array = np.empty((0, Board.columns), int)
        at_y = -1
        for y in range(Board.rows):
            if np.any(piece_candidates[y]):
                if at_y < 0:
                    at_y = y
                piece_array = np.append(piece_array, piece_candidates[y : y + 1], axis=0)

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


def _scan_image(num_rows: int, num_columns: int, image: np.ndarray, use_max: bool = False) -> np.ndarray:
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
            current_block = image[y_start_pos : y_end_pos + 1, x_start_pos : x_end_pos + 1]

            if use_max:
                # Use a 1-pixel inset to exclude boundary rows/columns that can
                # bleed bright pixels from adjacent filled cells into empty ones.
                inner = current_block[1:-1, 1:-1]
                sample = inner if inner.size > 0 else current_block
                filled = int(sample.max()) >= pixel_threshold_black
            else:
                # For the next-piece preview the block tiles are not grid-aligned,
                # so sampling the 2×2 center avoids false positives from tile-border
                # pixels that bleed into adjacent empty cells.
                block_middle = current_block[
                    round(block_height / 2) : round((block_height / 2) + 2),
                    round(block_width / 2) : round((block_width / 2) + 2),
                ]
                filled = not (block_middle < pixel_threshold_black).all()

            narray[y][x] = 1 if filled else 0

    return narray
