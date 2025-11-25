"""
Replay visualization system for viewing multiple training runs simultaneously.

Allows side-by-side comparison of bot performance across generations.
"""
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tetris.pieces import Tetrominoes
from tetris.recorder import GameRecording


class ReplayRenderer:
    """Renders a single game recording to a numpy array."""

    def __init__(
        self,
        block_size: int = 20,
        show_info: bool = True,
        show_next: bool = True,
    ):
        """
        Initialize renderer.

        Args:
            block_size: Size of each tetris block in pixels
            show_info: Show info panel with stats
            show_next: Show next piece
        """
        self.block_size = block_size
        self.show_info = show_info
        self.show_next = show_next

        # Colors for different pieces (BGR format for OpenCV)
        self.colors = {
            "i": (255, 0, 0),  # Blue
            "o": (0, 255, 255),  # Yellow
            "t": (255, 0, 255),  # Purple
            "s": (0, 255, 0),  # Green
            "z": (0, 0, 255),  # Red
            "j": (255, 128, 0),  # Orange
            "l": (255, 255, 255),  # White
        }

        # Calculate dimensions
        self.board_width = 10 * block_size
        self.board_height = 20 * block_size
        self.info_width = 120 if show_info else 0
        self.next_height = 80 if show_next else 0
        self.total_width = self.board_width + self.info_width
        self.total_height = self.board_height + self.next_height

    def render_snapshot(
        self, recording: GameRecording, snapshot_idx: int
    ) -> np.ndarray:
        """
        Render a specific snapshot to an image.

        Args:
            recording: Game recording
            snapshot_idx: Index of snapshot to render

        Returns:
            BGR image as numpy array
        """
        if snapshot_idx >= len(recording.snapshots):
            snapshot_idx = len(recording.snapshots) - 1

        snapshot = recording.snapshots[snapshot_idx]

        # Create blank canvas
        img = np.zeros((self.total_height, self.total_width, 3), dtype=np.uint8)

        # Draw board
        self._draw_board(img, snapshot.board_state)

        # Draw current piece
        self._draw_current_piece(
            img,
            snapshot.current_piece_name,
            snapshot.current_piece_rotation,
            snapshot.current_piece_x,
            snapshot.current_piece_y,
        )

        # Draw info panel
        if self.show_info:
            self._draw_info(img, recording, snapshot)

        # Draw next piece
        if self.show_next and snapshot.next_piece_name:
            self._draw_next_piece(img, snapshot.next_piece_name)

        return img

    def _draw_board(self, img: np.ndarray, board: np.ndarray):
        """Draw the board state."""
        for y in range(20):
            for x in range(10):
                color = (100, 100, 100) if board[y, x] != 0 else (20, 20, 20)
                x1 = x * self.block_size
                y1 = y * self.block_size
                x2 = x1 + self.block_size
                y2 = y1 + self.block_size

                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (50, 50, 50), 1)

    def _draw_current_piece(
        self,
        img: np.ndarray,
        piece_name: str,
        rotation: int,
        center_x: int,
        center_y: int,
    ):
        """Draw the current piece."""
        if not piece_name:
            return

        # Find the piece
        piece = next((p for p in Tetrominoes if p.name == piece_name), None)
        if not piece:
            return

        # Get shape
        shape = piece.shapes[rotation % len(piece.shapes)]
        color = self.colors.get(piece_name, (255, 255, 255))

        # Calculate corner position (0-based)
        corner_x = center_x - 2 - 1
        corner_y = center_y - 2 - 1

        # Draw each block of the piece
        for dy in range(5):
            for dx in range(5):
                if shape[dy, dx] != 0:
                    board_x = corner_x + dx
                    board_y = corner_y + dy

                    if 0 <= board_x < 10 and 0 <= board_y < 20:
                        x1 = board_x * self.block_size
                        y1 = board_y * self.block_size
                        x2 = x1 + self.block_size
                        y2 = y1 + self.block_size

                        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

    def _draw_info(
        self, img: np.ndarray, recording: GameRecording, snapshot
    ):
        """Draw info panel."""
        x_offset = self.board_width + 5
        y = 20

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        color = (255, 255, 255)
        thickness = 1

        # Generation info
        if recording.generation is not None:
            text = f"Gen:{recording.generation}"
            cv2.putText(img, text, (x_offset, y), font, font_scale, color, thickness)
            y += 20

        # Piece count
        text = f"Pcs:{snapshot.piece_number}"
        cv2.putText(img, text, (x_offset, y), font, font_scale, color, thickness)
        y += 20

        # Lines
        text = f"Lns:{snapshot.lines}"
        cv2.putText(img, text, (x_offset, y), font, font_scale, color, thickness)
        y += 20

        # Score
        text = f"Scr:{snapshot.score}"
        cv2.putText(img, text, (x_offset, y), font, font_scale, color, thickness)
        y += 20

        # Holes
        text = f"Hol:{snapshot.holes_count}"
        cv2.putText(img, text, (x_offset, y), font, font_scale, color, thickness)
        y += 20

        # Height
        text = f"Hgt:{snapshot.cumulative_height}"
        cv2.putText(img, text, (x_offset, y), font, font_scale, color, thickness)
        y += 20

        # Bumpiness
        text = f"Bmp:{snapshot.bumpiness}"
        cv2.putText(img, text, (x_offset, y), font, font_scale, color, thickness)
        y += 20

        # Move score
        if snapshot.move_score is not None:
            text = f"Mv:{snapshot.move_score:.1f}"
            cv2.putText(img, text, (x_offset, y), font, font_scale, color, thickness)
            y += 20

    def _draw_next_piece(self, img: np.ndarray, next_piece_name: str):
        """Draw next piece preview."""
        piece = next((p for p in Tetrominoes if p.name == next_piece_name), None)
        if not piece:
            return

        # Get default shape
        shape = piece.shapes[piece.default_shape_idx]
        color = self.colors.get(next_piece_name, (255, 255, 255))

        # Find non-zero bounds
        where = np.where(shape == 1)
        if len(where[0]) == 0:
            return

        min_y, max_y = np.min(where[0]), np.max(where[0])
        min_x, max_x = np.min(where[1]), np.max(where[1])

        # Center in next piece area
        piece_height = max_y - min_y + 1
        piece_width = max_x - min_x + 1

        # Calculate smaller block size for next piece
        small_block_size = self.block_size // 2
        y_offset = self.board_height + (self.next_height - piece_height * small_block_size) // 2
        x_offset = (self.board_width - piece_width * small_block_size) // 2

        # Draw piece
        for dy in range(min_y, max_y + 1):
            for dx in range(min_x, max_x + 1):
                if shape[dy, dx] != 0:
                    x1 = x_offset + (dx - min_x) * small_block_size
                    y1 = y_offset + (dy - min_y) * small_block_size
                    x2 = x1 + small_block_size
                    y2 = y1 + small_block_size

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)


class MultiReplayViewer:
    """View multiple game replays simultaneously in a grid."""

    def __init__(
        self,
        recordings: List[GameRecording],
        block_size: int = 15,
        grid_cols: int = 5,
        padding: int = 5,
    ):
        """
        Initialize multi-replay viewer.

        Args:
            recordings: List of game recordings to display
            block_size: Size of each block in pixels
            grid_cols: Number of columns in the grid
            padding: Padding between games in pixels
        """
        self.recordings = recordings
        self.renderer = ReplayRenderer(block_size=block_size)
        self.grid_cols = grid_cols
        self.padding = padding

        # Calculate grid dimensions
        self.grid_rows = (len(recordings) + grid_cols - 1) // grid_cols
        self.cell_width = self.renderer.total_width + padding
        self.cell_height = self.renderer.total_height + padding
        self.canvas_width = grid_cols * self.cell_width + padding
        self.canvas_height = self.grid_rows * self.cell_height + padding + 50  # Extra for controls

        self.current_frame = 0
        self.playing = False
        self.speed = 1.0

    def render_frame(self, frame_idx: int) -> np.ndarray:
        """
        Render all games at a specific frame.

        Args:
            frame_idx: Frame index to render

        Returns:
            Composite image with all games
        """
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)

        for idx, recording in enumerate(self.recordings):
            if not recording or len(recording.snapshots) == 0:
                continue

            row = idx // self.grid_cols
            col = idx % self.grid_cols

            x_offset = col * self.cell_width + self.padding
            y_offset = row * self.cell_height + self.padding

            # Find closest snapshot
            snapshot_idx = min(frame_idx, len(recording.snapshots) - 1)
            game_img = self.renderer.render_snapshot(recording, snapshot_idx)

            # Place in canvas
            canvas[
                y_offset : y_offset + game_img.shape[0],
                x_offset : x_offset + game_img.shape[1],
            ] = game_img

        # Draw controls at bottom
        self._draw_controls(canvas, frame_idx)

        return canvas

    def _draw_controls(self, canvas: np.ndarray, frame_idx: int):
        """Draw playback controls."""
        y = self.canvas_height - 35

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1

        # Frame info
        max_frames = max(len(r.snapshots) for r in self.recordings if r)
        text = f"Frame: {frame_idx}/{max_frames} | Speed: {self.speed}x | {'Playing' if self.playing else 'Paused'}"
        cv2.putText(canvas, text, (10, y), font, font_scale, color, thickness)

        # Instructions
        text = "Controls: SPACE=Play/Pause | LEFT/RIGHT=Step | +/-=Speed | Q=Quit"
        cv2.putText(canvas, text, (10, y + 20), font, 0.4, color, 1)

    def play(self, start_frame: int = 0, window_name: str = "Training Replay"):
        """
        Play the replays.

        Args:
            start_frame: Frame to start at
            window_name: OpenCV window name
        """
        self.current_frame = start_frame
        max_frames = max(len(r.snapshots) for r in self.recordings if r)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            # Render current frame
            canvas = self.render_frame(self.current_frame)
            cv2.imshow(window_name, canvas)

            # Handle keyboard input
            key = cv2.waitKey(int(16 / self.speed)) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                self.playing = not self.playing
            elif key == 81 or key == 2:  # Left arrow
                self.current_frame = max(0, self.current_frame - 1)
                self.playing = False
            elif key == 83 or key == 3:  # Right arrow
                self.current_frame = min(max_frames - 1, self.current_frame + 1)
                self.playing = False
            elif key == ord("+") or key == ord("="):
                self.speed = min(10.0, self.speed * 1.5)
            elif key == ord("-") or key == ord("_"):
                self.speed = max(0.1, self.speed / 1.5)
            elif key == ord("r"):
                self.current_frame = 0
                self.playing = True

            # Auto-advance if playing
            if self.playing:
                self.current_frame += 1
                if self.current_frame >= max_frames:
                    self.current_frame = 0  # Loop

        cv2.destroyAllWindows()


def view_generation_comparison(
    recording_dir: Path,
    generations: List[int],
    grid_cols: int = 5,
    block_size: int = 15,
):
    """
    View the best recording from multiple generations side-by-side.

    Args:
        recording_dir: Directory containing recordings
        generations: List of generation numbers to compare
        grid_cols: Number of columns in grid
        block_size: Size of blocks in pixels
    """
    from tetris.recorder import TrainingRecorder

    recorder = TrainingRecorder(recording_dir)
    recordings = []

    for gen in generations:
        files = recorder.get_generation_files(gen)
        if files:
            # Load all and find best
            gen_recordings = [GameRecording.load(f) for f in files]
            best = max(
                gen_recordings,
                key=lambda r: r.fitness if r.fitness is not None else r.final_lines,
            )
            recordings.append(best)
        else:
            recordings.append(None)

    viewer = MultiReplayViewer(recordings, block_size=block_size, grid_cols=grid_cols)
    viewer.play()


def view_generation_evolution(
    recording_dir: Path,
    max_generations: int,
    sample_every: int = 5,
    grid_cols: int = 5,
    block_size: int = 12,
):
    """
    View evolution of best recordings across generations.

    Args:
        recording_dir: Directory containing recordings
        max_generations: Maximum generation number to load
        sample_every: Show every Nth generation
        grid_cols: Number of columns in grid
        block_size: Size of blocks in pixels
    """
    from tetris.recorder import TrainingRecorder

    recorder = TrainingRecorder(recording_dir)

    # Load best from each generation
    all_recordings = recorder.load_best_per_generation(max_generations)

    # Sample
    recordings = [all_recordings[i] for i in range(0, len(all_recordings), sample_every)]

    # Filter out None values
    recordings = [r for r in recordings if r is not None]

    if not recordings:
        print("No recordings found!")
        return

    print(f"Viewing {len(recordings)} recordings from generations 0 to {max_generations}")
    viewer = MultiReplayViewer(recordings, block_size=block_size, grid_cols=grid_cols)
    viewer.play()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View training replay recordings")
    parser.add_argument(
        "recording_dir", type=Path, help="Directory containing recordings"
    )
    parser.add_argument(
        "--generations",
        type=int,
        nargs="+",
        help="Specific generations to view",
    )
    parser.add_argument(
        "--evolution",
        action="store_true",
        help="View evolution across all generations",
    )
    parser.add_argument(
        "--max-gen",
        type=int,
        default=100,
        help="Maximum generation for evolution view",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=5,
        help="Show every Nth generation in evolution view",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=5,
        help="Number of columns in grid",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=15,
        help="Size of tetris blocks in pixels",
    )

    args = parser.parse_args()

    if args.evolution:
        view_generation_evolution(
            args.recording_dir,
            args.max_gen,
            args.sample_every,
            args.grid_cols,
            args.block_size,
        )
    elif args.generations:
        view_generation_comparison(
            args.recording_dir,
            args.generations,
            args.grid_cols,
            args.block_size,
        )
    else:
        print("Please specify either --generations or --evolution")
