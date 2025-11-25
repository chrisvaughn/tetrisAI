"""
Game state recording for training visualization and analysis.

Records complete game states during training for later replay and analysis.
"""
import gzip
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .pieces import Piece
from .state import GameState


@dataclass
class GameSnapshot:
    """Snapshot of game state at a specific moment."""

    frame_number: int
    piece_number: int
    board_state: np.ndarray  # Will be RLE compressed when saved
    current_piece_name: str
    current_piece_rotation: int
    current_piece_x: int
    current_piece_y: int
    next_piece_name: Optional[str]
    score: int
    lines: int
    level: int
    holes_count: int
    cumulative_height: int
    bumpiness: int
    # Bot decision info
    move_rotations: Optional[int] = None
    move_translation: Optional[int] = None
    move_score: Optional[float] = None
    evaluation_time_ms: Optional[float] = None
    moves_considered: Optional[int] = None

    def compress_board(self):
        """Compress board state using run-length encoding."""
        # Simple RLE: store (value, count) pairs
        if len(self.board_state.shape) != 2:
            return self.board_state

        flat = self.board_state.flatten()
        compressed = []
        if len(flat) > 0:
            current_val = flat[0]
            count = 1
            for val in flat[1:]:
                if val == current_val and count < 255:
                    count += 1
                else:
                    compressed.append((current_val, count))
                    current_val = val
                    count = 1
            compressed.append((current_val, count))
        return compressed

    def decompress_board(self, compressed, shape=(20, 10)):
        """Decompress RLE board state."""
        if isinstance(compressed, np.ndarray):
            return compressed
        flat = []
        for val, count in compressed:
            flat.extend([val] * count)
        return np.array(flat, dtype=int).reshape(shape)


@dataclass
class GameRecording:
    """Complete recording of a single game."""

    game_id: str  # Unique identifier
    seed: int
    level: int
    bot_name: str
    generation: Optional[int] = None
    genome_id: Optional[int] = None
    fitness: Optional[float] = None
    snapshots: List[GameSnapshot] = field(default_factory=list)
    # Final statistics
    final_lines: int = 0
    final_score: int = 0
    final_pieces: int = 0
    piece_stats: dict = field(default_factory=dict)
    line_combos: dict = field(default_factory=dict)

    def add_snapshot(self, snapshot: GameSnapshot):
        """Add a snapshot to the recording."""
        self.snapshots.append(snapshot)

    def compress_snapshots(self):
        """Compress all board states in snapshots."""
        for snapshot in self.snapshots:
            if isinstance(snapshot.board_state, np.ndarray):
                snapshot.board_state = snapshot.compress_board()

    def save(self, filepath: Union[str, Path], compress=True):
        """Save recording to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Compress board states before saving
        if compress:
            self.compress_snapshots()

        # Save with gzip compression
        with gzip.open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Union[str, Path]) -> "GameRecording":
        """Load recording from file."""
        with gzip.open(filepath, "rb") as f:
            recording = pickle.load(f)

        # Decompress board states after loading
        for snapshot in recording.snapshots:
            if not isinstance(snapshot.board_state, np.ndarray):
                snapshot.board_state = snapshot.decompress_board(snapshot.board_state)

        return recording


class GameRecorder:
    """Records game states during gameplay for later analysis."""

    def __init__(
        self,
        game_id: str,
        seed: int,
        level: int,
        bot_name: str,
        generation: Optional[int] = None,
        genome_id: Optional[int] = None,
        snapshot_interval: int = 1,
    ):
        """
        Initialize recorder.

        Args:
            game_id: Unique identifier for this game
            seed: RNG seed used
            level: Starting level
            bot_name: Name of the bot playing
            generation: GA generation number (if applicable)
            genome_id: Genome ID (if applicable)
            snapshot_interval: Record every N pieces (1 = every piece)
        """
        self.recording = GameRecording(
            game_id=game_id,
            seed=seed,
            level=level,
            bot_name=bot_name,
            generation=generation,
            genome_id=genome_id,
        )
        self.snapshot_interval = snapshot_interval
        self.frame_counter = 0
        self.piece_counter = 0
        self.last_snapshot_piece = -1

    def should_record(self) -> bool:
        """Check if we should record current state."""
        return (
            self.piece_counter - self.last_snapshot_piece >= self.snapshot_interval
        )

    def record_state(
        self,
        game_state: GameState,
        score: int,
        lines: int,
        level: int,
        move_info: Optional[dict] = None,
    ):
        """
        Record current game state.

        Args:
            game_state: Current GameState object
            score: Current score
            lines: Lines cleared so far
            level: Current level
            move_info: Optional dict with bot move information:
                - rotations: number of rotations
                - translation: horizontal translation
                - score: move evaluation score
                - evaluation_time_ms: time to evaluate
                - moves_considered: number of moves evaluated
        """
        if not self.should_record():
            return

        # Calculate board metrics
        holes_count, _ = game_state.count_holes()
        cumulative_height = game_state.cumulative_height()
        bumpiness = game_state.roughness()

        # Extract piece information
        current_piece = game_state.current_piece
        next_piece = game_state.next_piece

        snapshot = GameSnapshot(
            frame_number=self.frame_counter,
            piece_number=self.piece_counter,
            board_state=game_state.board.board.copy(),
            current_piece_name=current_piece.name if current_piece else "",
            current_piece_rotation=(
                current_piece.current_shape_idx if current_piece else 0
            ),
            current_piece_x=current_piece._x if current_piece else 0,
            current_piece_y=current_piece._y if current_piece else 0,
            next_piece_name=next_piece.name if next_piece else None,
            score=score,
            lines=lines,
            level=level,
            holes_count=holes_count,
            cumulative_height=cumulative_height,
            bumpiness=bumpiness,
        )

        # Add move information if provided
        if move_info:
            snapshot.move_rotations = move_info.get("rotations")
            snapshot.move_translation = move_info.get("translation")
            snapshot.move_score = move_info.get("score")
            snapshot.evaluation_time_ms = move_info.get("evaluation_time_ms")
            snapshot.moves_considered = move_info.get("moves_considered")

        self.recording.add_snapshot(snapshot)
        self.last_snapshot_piece = self.piece_counter

    def increment_frame(self):
        """Increment frame counter."""
        self.frame_counter += 1

    def increment_piece(self):
        """Increment piece counter."""
        self.piece_counter += 1

    def finalize(
        self, final_lines: int, final_score: int, piece_stats: dict, line_combos: dict
    ):
        """
        Finalize recording with final statistics.

        Args:
            final_lines: Total lines cleared
            final_score: Final score
            piece_stats: Dictionary of piece counts
            line_combos: Dictionary of line combo counts
        """
        self.recording.final_lines = final_lines
        self.recording.final_score = final_score
        self.recording.final_pieces = self.piece_counter
        self.recording.piece_stats = dict(piece_stats)
        self.recording.line_combos = dict(line_combos)

    def save(self, filepath: Union[str, Path], compress=True):
        """Save recording to file."""
        self.recording.save(filepath, compress=compress)

    def get_recording(self) -> GameRecording:
        """Get the recording object."""
        return self.recording


class TrainingRecorder:
    """Manages recording of multiple games during training."""

    def __init__(self, output_dir: Union[str, Path], snapshot_interval: int = 1):
        """
        Initialize training recorder.

        Args:
            output_dir: Directory to save recordings
            snapshot_interval: Record every N pieces
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = snapshot_interval
        self.recordings: List[GameRecording] = []

    def create_recorder(
        self,
        game_id: str,
        seed: int,
        level: int,
        bot_name: str,
        generation: Optional[int] = None,
        genome_id: Optional[int] = None,
    ) -> GameRecorder:
        """Create a new game recorder."""
        return GameRecorder(
            game_id=game_id,
            seed=seed,
            level=level,
            bot_name=bot_name,
            generation=generation,
            genome_id=genome_id,
            snapshot_interval=self.snapshot_interval,
        )

    def save_recorder(self, recorder: GameRecorder, generation: int, genome_id: int):
        """Save a completed recording."""
        filename = f"gen_{generation:04d}_genome_{genome_id:04d}.pkl.gz"
        filepath = self.output_dir / filename
        recorder.save(filepath)
        self.recordings.append(recorder.get_recording())

    def get_generation_files(self, generation: int) -> List[Path]:
        """Get all recording files for a specific generation."""
        pattern = f"gen_{generation:04d}_*.pkl.gz"
        return sorted(self.output_dir.glob(pattern))

    def load_generation_recordings(self, generation: int) -> List[GameRecording]:
        """Load all recordings from a specific generation."""
        files = self.get_generation_files(generation)
        return [GameRecording.load(f) for f in files]

    def load_best_per_generation(
        self, max_generations: int
    ) -> List[Optional[GameRecording]]:
        """
        Load the best recording from each generation.

        Returns a list where index = generation number.
        """
        recordings = []
        for gen in range(max_generations):
            files = self.get_generation_files(gen)
            if not files:
                recordings.append(None)
                continue

            # Load all recordings and find the best
            gen_recordings = [GameRecording.load(f) for f in files]
            best = max(
                gen_recordings,
                key=lambda r: r.fitness if r.fitness is not None else r.final_lines,
            )
            recordings.append(best)

        return recordings
