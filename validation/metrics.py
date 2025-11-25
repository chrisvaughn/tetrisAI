"""
Validation metrics for training analysis.

Provides tools to analyze training progress, detect convergence,
and compare bot performance across generations.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from tetris.recorder import GameRecording


@dataclass
class GenerationMetrics:
    """Metrics for a single generation."""

    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    best_lines: int
    avg_lines: float
    best_score: int
    avg_score: float
    avg_pieces: float
    avg_holes: float
    avg_height: float
    avg_bumpiness: float
    tetris_rate: float  # Percentage of 4-line clears
    move_diversity: float  # Entropy of move selections
    avg_evaluation_time_ms: float


@dataclass
class TrainingAnalysis:
    """Complete analysis of training progress."""

    generations: List[GenerationMetrics] = field(default_factory=list)

    def add_generation(self, metrics: GenerationMetrics):
        """Add metrics for a generation."""
        self.generations.append(metrics)

    def is_converged(
        self, window: int = 10, threshold: float = 0.01
    ) -> bool:
        """
        Check if training has converged.

        Args:
            window: Number of recent generations to check
            threshold: Maximum relative change to consider converged

        Returns:
            True if converged
        """
        if len(self.generations) < window:
            return False

        recent = self.generations[-window:]
        fitnesses = [g.best_fitness for g in recent]

        if min(fitnesses) == 0:
            return False

        # Calculate coefficient of variation
        mean = np.mean(fitnesses)
        std = np.std(fitnesses)
        cv = std / mean

        return cv < threshold

    def get_convergence_generation(
        self, window: int = 10, threshold: float = 0.01
    ) -> Optional[int]:
        """
        Find the generation where training converged.

        Returns generation number or None if not converged.
        """
        for i in range(window, len(self.generations)):
            subset = self.generations[: i + 1]
            recent = subset[-window:]
            fitnesses = [g.best_fitness for g in recent]

            if min(fitnesses) > 0:
                mean = np.mean(fitnesses)
                std = np.std(fitnesses)
                cv = std / mean

                if cv < threshold:
                    return i - window + 1

        return None

    def get_improvement_rate(self, window: int = 10) -> List[float]:
        """
        Calculate rate of improvement over time.

        Returns list of improvement rates per generation.
        """
        if len(self.generations) < 2:
            return []

        rates = []
        for i in range(1, len(self.generations)):
            prev = self.generations[max(0, i - window)]
            curr = self.generations[i]

            if prev.best_fitness > 0:
                rate = (curr.best_fitness - prev.best_fitness) / prev.best_fitness
            else:
                rate = 0.0

            rates.append(rate)

        return rates

    def get_plateaus(
        self, min_length: int = 5, threshold: float = 0.001
    ) -> List[tuple[int, int]]:
        """
        Find plateaus in training progress.

        Args:
            min_length: Minimum plateau length
            threshold: Maximum change to consider a plateau

        Returns:
            List of (start_gen, end_gen) tuples
        """
        if len(self.generations) < min_length:
            return []

        plateaus = []
        plateau_start = None

        for i in range(1, len(self.generations)):
            prev = self.generations[i - 1].best_fitness
            curr = self.generations[i].best_fitness

            if prev > 0:
                change = abs(curr - prev) / prev
            else:
                change = float("inf")

            if change < threshold:
                if plateau_start is None:
                    plateau_start = i - 1
            else:
                if plateau_start is not None:
                    length = i - plateau_start
                    if length >= min_length:
                        plateaus.append((plateau_start, i - 1))
                    plateau_start = None

        # Check for plateau at end
        if plateau_start is not None:
            length = len(self.generations) - plateau_start
            if length >= min_length:
                plateaus.append((plateau_start, len(self.generations) - 1))

        return plateaus

    def print_summary(self):
        """Print summary of training progress."""
        if not self.generations:
            print("No generations recorded")
            return

        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        first = self.generations[0]
        last = self.generations[-1]

        print(f"\nGenerations: {len(self.generations)}")
        print(f"\nFitness Progress:")
        print(f"  Initial best: {first.best_fitness:.2f}")
        print(f"  Final best:   {last.best_fitness:.2f}")
        print(
            f"  Improvement:  {last.best_fitness - first.best_fitness:.2f} ({(last.best_fitness/first.best_fitness - 1)*100:.1f}%)"
        )

        print(f"\nLines Cleared:")
        print(f"  Initial best: {first.best_lines}")
        print(f"  Final best:   {last.best_lines}")
        print(f"  Improvement:  {last.best_lines - first.best_lines}")

        print(f"\nScore:")
        print(f"  Initial best: {first.best_score:,}")
        print(f"  Final best:   {last.best_score:,}")

        print(f"\nBoard Metrics (final generation averages):")
        print(f"  Avg holes:     {last.avg_holes:.1f}")
        print(f"  Avg height:    {last.avg_height:.1f}")
        print(f"  Avg bumpiness: {last.avg_bumpiness:.1f}")
        print(f"  Tetris rate:   {last.tetris_rate:.1%}")

        # Convergence analysis
        if self.is_converged():
            conv_gen = self.get_convergence_generation()
            if conv_gen is not None:
                print(f"\nConvergence: Generation {conv_gen}")
        else:
            print("\nConvergence: Not yet converged")

        # Plateaus
        plateaus = self.get_plateaus()
        if plateaus:
            print(f"\nPlateaus detected: {len(plateaus)}")
            for start, end in plateaus[:3]:  # Show first 3
                print(f"  Generations {start}-{end} (length: {end-start+1})")

        print("=" * 60 + "\n")


def calculate_generation_metrics(
    recordings: List[GameRecording], generation: int
) -> GenerationMetrics:
    """
    Calculate metrics for a generation from recordings.

    Args:
        recordings: List of game recordings from this generation
        generation: Generation number

    Returns:
        GenerationMetrics object
    """
    if not recordings:
        return GenerationMetrics(
            generation=generation,
            best_fitness=0,
            avg_fitness=0,
            worst_fitness=0,
            best_lines=0,
            avg_lines=0,
            best_score=0,
            avg_score=0,
            avg_pieces=0,
            avg_holes=0,
            avg_height=0,
            avg_bumpiness=0,
            tetris_rate=0,
            move_diversity=0,
            avg_evaluation_time_ms=0,
        )

    # Extract basic stats
    fitnesses = [r.fitness if r.fitness is not None else r.final_lines for r in recordings]
    lines = [r.final_lines for r in recordings]
    scores = [r.final_score for r in recordings]
    pieces = [r.final_pieces for r in recordings]

    # Calculate average board metrics from snapshots
    all_holes = []
    all_heights = []
    all_bumpiness = []
    all_eval_times = []
    tetris_count = 0
    total_line_clears = 0

    # Move diversity - track unique move patterns
    move_patterns: Dict[tuple, int] = {}

    for recording in recordings:
        # Collect metrics from snapshots
        for snapshot in recording.snapshots:
            all_holes.append(snapshot.holes_count)
            all_heights.append(snapshot.cumulative_height)
            all_bumpiness.append(snapshot.bumpiness)

            if snapshot.evaluation_time_ms is not None:
                all_eval_times.append(snapshot.evaluation_time_ms)

            # Track move patterns for diversity
            if (
                snapshot.move_rotations is not None
                and snapshot.move_translation is not None
            ):
                pattern = (snapshot.move_rotations, snapshot.move_translation)
                move_patterns[pattern] = move_patterns.get(pattern, 0) + 1

        # Count tetris (4-line clears)
        if hasattr(recording, "line_combos"):
            tetris_count += recording.line_combos.get(4, 0)
            total_line_clears += sum(recording.line_combos.values())

    # Calculate move diversity (entropy)
    if move_patterns:
        total_moves = sum(move_patterns.values())
        probabilities = [count / total_moves for count in move_patterns.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(move_patterns))
        move_diversity = entropy / max_entropy if max_entropy > 0 else 0
    else:
        move_diversity = 0

    # Calculate tetris rate
    tetris_rate = tetris_count / total_line_clears if total_line_clears > 0 else 0

    return GenerationMetrics(
        generation=generation,
        best_fitness=max(fitnesses),
        avg_fitness=np.mean(fitnesses),
        worst_fitness=min(fitnesses),
        best_lines=max(lines),
        avg_lines=np.mean(lines),
        best_score=max(scores),
        avg_score=np.mean(scores),
        avg_pieces=np.mean(pieces),
        avg_holes=np.mean(all_holes) if all_holes else 0,
        avg_height=np.mean(all_heights) if all_heights else 0,
        avg_bumpiness=np.mean(all_bumpiness) if all_bumpiness else 0,
        tetris_rate=tetris_rate,
        move_diversity=move_diversity,
        avg_evaluation_time_ms=np.mean(all_eval_times) if all_eval_times else 0,
    )


def analyze_training(recording_dir: Path, max_generations: int) -> TrainingAnalysis:
    """
    Analyze complete training run from recordings.

    Args:
        recording_dir: Directory containing recordings
        max_generations: Maximum generation to analyze

    Returns:
        TrainingAnalysis object
    """
    from tetris.recorder import TrainingRecorder

    recorder = TrainingRecorder(recording_dir)
    analysis = TrainingAnalysis()

    for gen in range(max_generations):
        files = recorder.get_generation_files(gen)
        if not files:
            continue

        recordings = [GameRecording.load(f) for f in files]
        metrics = calculate_generation_metrics(recordings, gen)
        analysis.add_generation(metrics)

    return analysis


def compare_runs(
    run_dirs: List[Path], labels: Optional[List[str]] = None
) -> Dict[str, TrainingAnalysis]:
    """
    Compare multiple training runs.

    Args:
        run_dirs: List of recording directories
        labels: Optional labels for each run

    Returns:
        Dictionary mapping label to TrainingAnalysis
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(run_dirs))]

    results = {}
    for run_dir, label in zip(run_dirs, labels):
        # Find max generation
        from tetris.recorder import TrainingRecorder

        recorder = TrainingRecorder(run_dir)
        max_gen = 0
        while recorder.get_generation_files(max_gen):
            max_gen += 1

        analysis = analyze_training(run_dir, max_gen)
        results[label] = analysis

    return results


def export_metrics_csv(analysis: TrainingAnalysis, output_path: Path):
    """Export metrics to CSV for external analysis."""
    import csv

    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "generation",
            "best_fitness",
            "avg_fitness",
            "worst_fitness",
            "best_lines",
            "avg_lines",
            "best_score",
            "avg_score",
            "avg_pieces",
            "avg_holes",
            "avg_height",
            "avg_bumpiness",
            "tetris_rate",
            "move_diversity",
            "avg_evaluation_time_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for metrics in analysis.generations:
            writer.writerow(
                {
                    "generation": metrics.generation,
                    "best_fitness": metrics.best_fitness,
                    "avg_fitness": metrics.avg_fitness,
                    "worst_fitness": metrics.worst_fitness,
                    "best_lines": metrics.best_lines,
                    "avg_lines": metrics.avg_lines,
                    "best_score": metrics.best_score,
                    "avg_score": metrics.avg_score,
                    "avg_pieces": metrics.avg_pieces,
                    "avg_holes": metrics.avg_holes,
                    "avg_height": metrics.avg_height,
                    "avg_bumpiness": metrics.avg_bumpiness,
                    "tetris_rate": metrics.tetris_rate,
                    "move_diversity": metrics.move_diversity,
                    "avg_evaluation_time_ms": metrics.avg_evaluation_time_ms,
                }
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze training metrics")
    parser.add_argument("recording_dir", type=Path, help="Recording directory")
    parser.add_argument(
        "--max-gen", type=int, default=100, help="Maximum generation to analyze"
    )
    parser.add_argument(
        "--export-csv", type=Path, help="Export metrics to CSV file"
    )

    args = parser.parse_args()

    print(f"Analyzing training run in {args.recording_dir}...")
    analysis = analyze_training(args.recording_dir, args.max_gen)
    analysis.print_summary()

    if args.export_csv:
        export_metrics_csv(analysis, args.export_csv)
        print(f"\nMetrics exported to {args.export_csv}")
