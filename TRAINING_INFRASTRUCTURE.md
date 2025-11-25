# Training Infrastructure Guide

This guide covers the new training infrastructure for recording, replaying, and analyzing Tetris bot training runs.

## Overview

The training infrastructure provides:

1. **Game State Recording** - Capture full game states during training
2. **Replay Visualization** - View multiple training runs side-by-side
3. **Validation Metrics** - Analyze training progress and convergence
4. **NES Accuracy Audit** - Documentation of implementation fidelity

## Quick Start

### 1. Train with Recording

```bash
# Basic training with recording
python train_with_recording.py --fitness lines --population 100 --generations 50

# With custom settings
python train_with_recording.py \
  --fitness score \
  --population 50 \
  --generations 100 \
  --recording-dir ./my_recordings \
  --snapshot-interval 5 \
  --parallel-runners 8
```

**Parameters:**
- `--recording-dir`: Where to save recordings (default: `./recordings`)
- `--snapshot-interval`: Record every N pieces (1=every piece, 5=every 5th piece)
- `--resume`: Resume from existing save file
- Other parameters same as `train.py`

### 2. View Training Replays

```bash
# View evolution across all generations (samples every 5th)
python visualization/replay_viewer.py ./recordings --evolution --max-gen 50

# View specific generations side-by-side
python visualization/replay_viewer.py ./recordings --generations 0 10 20 30 40 50

# Customize display
python visualization/replay_viewer.py ./recordings \
  --evolution \
  --max-gen 100 \
  --sample-every 10 \
  --grid-cols 4 \
  --block-size 12
```

**Controls during playback:**
- `SPACE` - Play/Pause
- `LEFT/RIGHT` - Step backward/forward
- `+/-` - Increase/decrease speed
- `R` - Reset to beginning
- `Q` - Quit

### 3. Analyze Training Metrics

```bash
# Print training summary
python validation/metrics.py ./recordings --max-gen 50

# Export to CSV for plotting
python validation/metrics.py ./recordings --max-gen 50 --export-csv metrics.csv
```

## Architecture

### Recording System

**Files:**
- `tetris/recorder.py` - Core recording infrastructure
- `train_with_recording.py` - Enhanced training script

**Key Classes:**

```python
# Snapshot of game state at one moment
@dataclass
class GameSnapshot:
    frame_number: int
    board_state: np.ndarray
    score: int
    lines: int
    holes_count: int
    cumulative_height: int
    bumpiness: int
    move_info: dict  # Bot's decision details

# Complete recording of one game
@dataclass
class GameRecording:
    game_id: str
    generation: int
    genome_id: int
    fitness: float
    snapshots: List[GameSnapshot]
    final_lines: int
    final_score: int

# Manages recording during gameplay
class GameRecorder:
    def record_state(self, game_state, score, lines, move_info)
    def save(self, filepath)

# Manages recordings across training
class TrainingRecorder:
    def create_recorder(self, game_id, seed, level, bot_name, generation, genome_id)
    def save_recorder(self, recorder, generation, genome_id)
    def load_generation_recordings(self, generation)
```

**Storage Format:**
- Gzip-compressed pickle files
- Board states use run-length encoding
- One file per game: `gen_XXXX_genome_YYYY.pkl.gz`
- Typical size: 50-200 KB per game

### Visualization System

**Files:**
- `visualization/replay_viewer.py` - Multi-game replay viewer

**Key Classes:**

```python
# Renders single game to image
class ReplayRenderer:
    def render_snapshot(self, recording, snapshot_idx) -> np.ndarray

# Views multiple games in grid
class MultiReplayViewer:
    def __init__(self, recordings, block_size, grid_cols)
    def play(self, start_frame, window_name)

# Convenience functions
def view_generation_comparison(recording_dir, generations, grid_cols)
def view_generation_evolution(recording_dir, max_generations, sample_every)
```

### Validation System

**Files:**
- `validation/metrics.py` - Training analysis metrics

**Key Classes:**

```python
@dataclass
class GenerationMetrics:
    best_fitness: float
    avg_fitness: float
    best_lines: int
    avg_lines: float
    avg_holes: float
    tetris_rate: float
    move_diversity: float
    # ... many more metrics

class TrainingAnalysis:
    def is_converged(self, window, threshold) -> bool
    def get_convergence_generation() -> int
    def get_improvement_rate() -> List[float]
    def get_plateaus() -> List[tuple[int, int]]
    def print_summary()

# Functions
def calculate_generation_metrics(recordings, generation) -> GenerationMetrics
def analyze_training(recording_dir, max_generations) -> TrainingAnalysis
def compare_runs(run_dirs, labels) -> Dict[str, TrainingAnalysis]
def export_metrics_csv(analysis, output_path)
```

## Usage Examples

### Example 1: Quick Training Session

```bash
# Train for 20 generations with recording
python train_with_recording.py --generations 20 --population 50

# View the evolution
python visualization/replay_viewer.py ./recordings --evolution --max-gen 20

# Analyze results
python validation/metrics.py ./recordings --max-gen 20
```

### Example 2: Long Training with Analysis

```bash
# Start long training run
python train_with_recording.py \
  --generations 100 \
  --population 100 \
  --recording-dir ./long_run \
  --snapshot-interval 10  # Save space by recording every 10th piece

# Later, view key generations
python visualization/replay_viewer.py ./long_run \
  --generations 0 25 50 75 100 \
  --grid-cols 5

# Export metrics for plotting
python validation/metrics.py ./long_run --max-gen 100 --export-csv long_run.csv

# Plot in Python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('long_run.csv')
plt.plot(df['generation'], df['best_fitness'])
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Training Progress')
plt.show()
```

### Example 3: Compare Training Runs

```python
from pathlib import Path
from validation import compare_runs

# Compare different hyperparameters
results = compare_runs(
    run_dirs=[
        Path('./run_pop50'),
        Path('./run_pop100'),
        Path('./run_pop200'),
    ],
    labels=['Pop 50', 'Pop 100', 'Pop 200']
)

for label, analysis in results.items():
    print(f"\n{label}:")
    print(f"  Final fitness: {analysis.generations[-1].best_fitness}")
    print(f"  Converged at gen: {analysis.get_convergence_generation()}")
```

### Example 4: Programmatic Replay

```python
from pathlib import Path
from tetris.recorder import GameRecording
from visualization import ReplayRenderer

# Load specific game
recording = GameRecording.load(Path('./recordings/gen_0050_genome_0001.pkl.gz'))

# Create renderer
renderer = ReplayRenderer(block_size=25, show_info=True)

# Render specific frame
img = renderer.render_snapshot(recording, snapshot_idx=100)

# Save as image or process further
import cv2
cv2.imwrite('snapshot.png', img)
```

## Storage Considerations

### Disk Space

Recording every piece with full state:
- ~50-200 KB per game
- Population of 100: ~5-20 MB per generation
- 100 generations: ~0.5-2 GB total

To reduce storage:
1. Increase `--snapshot-interval` (record every 5-10 pieces)
2. Record only best genomes (modify `train_with_recording.py`)
3. Compress older generations more aggressively
4. Delete intermediate generations after training

### Example: Minimal Recording

```python
# In train_with_recording.py, modify to only record best:

def fitness_with_recording(weights):
    # ... evaluate ...

    # Only record if this is the best so far
    if fitness > best_fitness_so_far[0]:
        # Create and save recording
        best_fitness_so_far[0] = fitness
```

## Integration with Existing Code

The recording system is designed to be **optional and non-invasive**:

- Use `train.py` for normal training (no recording overhead)
- Use `train_with_recording.py` when you want to capture data
- Both use the same GA and bot code
- Recording adds ~5-10% training time overhead

## Metrics Explained

### Fitness Metrics
- **best_fitness**: Highest fitness in generation
- **avg_fitness**: Average fitness across population
- **worst_fitness**: Lowest fitness in generation

### Game Performance
- **best_lines**: Most lines cleared by any genome
- **avg_lines**: Average lines cleared
- **best_score**: Highest score achieved
- **avg_score**: Average score

### Board Quality
- **avg_holes**: Average holes in board (lower is better)
- **avg_height**: Average cumulative height (varies by strategy)
- **avg_bumpiness**: Average surface roughness (lower is usually better)

### Strategy Metrics
- **tetris_rate**: % of line clears that are 4-liners (higher is more efficient)
- **move_diversity**: Entropy of move choices (higher = more varied strategy)
- **avg_evaluation_time_ms**: Time to evaluate moves (for performance tuning)

### Training Progress
- **convergence**: When fitness stabilizes (based on coefficient of variation)
- **improvement_rate**: % fitness improvement per generation
- **plateaus**: Periods of little fitness improvement

## Troubleshooting

### Recording files too large

```bash
# Increase snapshot interval
python train_with_recording.py --snapshot-interval 10

# Or reduce number of iterations
python train_with_recording.py --num-iterations 50
```

### Replay viewer crashes or slow

```bash
# Reduce block size
python visualization/replay_viewer.py ./recordings --evolution --block-size 10

# Show fewer games at once
python visualization/replay_viewer.py ./recordings --evolution --grid-cols 3

# Sample less frequently
python visualization/replay_viewer.py ./recordings --evolution --sample-every 10
```

### Out of memory during analysis

```python
# Analyze in batches
from validation import analyze_training

for start in range(0, 100, 10):
    analysis = analyze_training(recording_dir, max_generations=start+10)
    # Process this batch
```

## Next Steps

### Before Neural Network Training

Based on the audit in `NES_TETRIS_AUDIT.md`, consider:

1. **Verify rotation system** - Create tests comparing to NES behavior
2. **Implement wall kicks** - Add basic wall kick mechanics
3. **Test on emulator** - Validate trained bots work on real NES
4. **Baseline performance** - Record GA-trained WeightedBot performance as baseline

### Extending the System

Ideas for enhancements:

1. **Real-time monitoring** - Web dashboard showing training progress
2. **Automatic best selection** - Auto-pick best recordings for visualization
3. **Heatmaps** - Visualize where pieces are placed over time
4. **Move analysis** - Detailed breakdown of move patterns
5. **A/B testing** - Statistical comparison of training configurations

## References

- `NES_TETRIS_AUDIT.md` - Detailed audit of implementation accuracy
- `CLAUDE.md` - General project documentation
- `tetris/recorder.py` - Recording implementation
- `visualization/replay_viewer.py` - Visualization implementation
- `validation/metrics.py` - Metrics implementation
