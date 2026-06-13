"""Live cv2 window showing the bot's current view of the board.

Works for both in-memory and emulator runs since both produce a GameState.
"""

import time

import cv2

from tetris.state import GameState

from .board_renderer import BoardRenderer


class LiveViewer:
    """cv2 window that shows the bot's current view of the board.

    Redraws are throttled to `max_fps` so high-frequency callers (e.g. the
    emulator capture loop) don't stall on cv2 window updates.
    """

    def __init__(self, window_name: str = "Bot View", renderer: BoardRenderer = None, max_fps: float = 30):
        self.window_name = window_name
        self.renderer = renderer or BoardRenderer()
        self.min_frame_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._last_draw_time = 0.0
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def update(
        self,
        state: GameState,
        best_move=None,
        info: dict = None,
    ) -> bool:
        """Render state and pump the cv2 event loop.

        `best_move`, if provided, overlays its expected end-state as a ghost
        placement. Returns False if the user pressed 'q' (caller should stop).
        """
        now = time.time()
        if now - self._last_draw_time < self.min_frame_interval:
            return True
        self._last_draw_time = now

        ghost_state = best_move.end_state if best_move is not None else None
        img = self.renderer.render(state, ghost_state=ghost_state, info=info)
        cv2.imshow(self.window_name, img)

        return cv2.waitKey(1) & 0xFF != ord("q")

    def close(self):
        cv2.destroyWindow(self.window_name)
