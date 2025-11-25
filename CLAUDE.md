# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Tetris AI project implementing bot strategies using Genetic Algorithms (GA) with weighted evaluation functions. The project includes a complete Tetris game engine, bot framework, and GA-based training system for evolving optimal play strategies.

## Environment Setup

### Python Environment
```bash
# Setup with pyenv
export py_version=3.12
pyenv install $py_version --skip-existing
pyenv uninstall -f tetrisAI
pyenv virtualenv $py_version tetrisAI
pyenv local tetrisAI
```

### Dependencies
```bash
# For macOS with emulator support
poetry install -E macOS

# For Linux training environment (no emulator)
poetry install
```

## Running the Code

### Running Bots

**In-memory simulation (fast, for training/testing):**
```bash
python run.py --bot-model WeightedBotLines
python run.py --bot-model Random --seed 12345 --stats
```

**With emulator (plays actual NES Tetris):**
```bash
python run.py --emulator --bot-model WeightedBotLines --stats
```

**Using saved weights from training:**
```bash
python run.py --bot-model WeightedBotScore --save-file save_score.pkl --save-gen 50
```

**Common flags:**
- `--stats`: Print move statistics during gameplay
- `--drop`: Enable soft drop (pieces fall faster)
- `--seed <int>`: Set RNG seed for reproducibility
- `--level <int>`: Starting level (default: 19)
- `--debug`: Enable detailed debug logging

### Training Bots

**Genetic Algorithm training (WeightedBot):**
```bash
# Train for lines optimization
python train.py --fitness lines --population 100 --generations 100

# Train for score optimization
python train.py --fitness score --population 100 --generations 50 --save-file custom_save.pkl

# Control parallelism
python train.py --parallel-runners 8
python train.py --no-parallel  # Disable parallel evaluation
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/tetris/test_board.py

# Run with coverage
pytest --cov=tetris --cov=bot
```

### Code Formatting

```bash
# Format code
black .

# Sort imports
isort .

# Run both (common workflow)
isort . && black .
```

## Architecture

### Core Game Engine (`tetris/`)

**Game Loop Flow:**
1. Game spawns new piece
2. Bot receives state update via `bot.update_state(game.state)`
3. Bot evaluates possible moves and returns best move
4. Move sequence is converted to actions and executed
5. Piece locks, lines clear, new piece spawns
6. Repeat until game over

**Key Classes:**
- **Game** (`game.py`): Main game controller with threading for continuous play
- **GameState** (`state.py`): Represents game state at any moment; cloneable for move simulation
- **Board** (`board.py`): 10x20 grid with line clearing and collision detection
- **Piece** (`pieces.py`): Tetromino pieces with rotation and movement logic

**NES Tetris PRNG:** Uses authentic NES piece generation algorithm (`nes_prng()`) for reproducible piece sequences.

### Bot System (`bot/`)

All bots inherit from `BaseBot` and implement:
- `update_state(state)`: Receive current game state
- `get_best_move(debug)`: Return `BotMove` with rotations and translations
- `evaluate_move(rotations, translation)`: Score individual moves

**BotMove Structure:**
- `rotations` (0-3): Number of clockwise rotations
- `translation` (int): Horizontal movement (negative=left, positive=right)
- `score`: Evaluation score
- `end_state`: Expected game state after move
- `lines_completed`: Lines cleared by this move
- `to_sequence()`: Converts to list of action tuples

**Bot Implementations:**

1. **WeightedBot** (`bot/weighted_bot/`):
   - Uses weighted evaluation functions (holes, height, lines, bumpiness, etc.)
   - Evaluates all possible moves by simulating end states
   - Weights evolved via GA in `train.py`
   - Supports parallel evaluation via multiprocessing pool
   - Evaluation functions in `evaluate.py`, predefined weights in `defined_weights.py`

2. **RandomBot** (`bot/random_bot/`):
   - Random move selection for baseline comparison

### Training System

**Genetic Algorithm** (`train.py`):
- Evolves weights for WeightedBot evaluation functions
- Fitness methods: "lines" (maximize lines cleared) or "score" (maximize points)
- Process: Initialize population → Evaluate → Select → Crossover/Mutate → Repeat
- Saves progress in pickle files (can resume with `resume=True`)
- Evaluation runs multiple games per genome and averages top 1/3 results
- Evolution logic in `bot/weighted_bot/evolution.py` with tournament selection, crossover, and mutation

### Vision System (`vision/`)

**Detectorist** (`detect.py`):
- Computer vision for real emulator integration
- Detects board state, current piece, and next piece from screen captures
- Works with 256x240 NES resolution
- Emulator control via `emulator/` package (macOS-specific)

## Development Notes

### Move Execution Pattern

The standard pattern for executing bot moves:
```python
if game.state.new_piece():
    bot.update_state(game.state)
    best_move, time_taken, moves_considered = bot.get_best_move(debug=False)
    move_sequence = best_move.to_sequence()

while move_sequence:
    moves = move_sequence.pop(0)
    for move in moves:
        if move != "noop":
            getattr(game, move)()
    game.move_seq_complete()
```

### State Cloning for Move Simulation

WeightedBot simulates moves by cloning state:
```python
test_state = current_state.clone()
execute_move(test_state, rotations, translation)
score, _ = evaluator.scoring_func(test_state)
```

This allows evaluating all possible moves without affecting game state.

### Parallel Evaluation

WeightedBot uses multiprocessing pool for evaluating moves:
- Pool initialized via `get_pool()` in `evaluation_pool.py`
- Distributes move evaluation across CPU cores
- Significant speedup for training and gameplay

### Emulator Integration

With `--emulator` flag, the system:
1. Captures screen from NES emulator
2. Uses vision system to detect board and pieces
3. Sends keyboard commands to emulator
4. Plans next move while current piece is falling (optimization)

### Scoring Versions

Two scoring systems available via `--scoring` flag:
- `v1`: Original scoring
- `v2`: Updated scoring system (default)

Used consistently across training and evaluation for fair comparison.

## Common Development Tasks

### Adding a New Bot Type

1. Create subclass of `BaseBot` in `bot/your_bot/`
2. Implement required methods: `update_state()`, `get_best_move()`, `evaluate_move()`
3. Return `BotMove` objects from `get_best_move()`
4. Add to `run.py` bot selection logic
5. Test with `python run.py --bot-model YourBot`

### Modifying Evaluation Functions

For WeightedBot, edit `bot/weighted_bot/evaluate.py`:
- Add new heuristic function
- Add corresponding weight to `Weights` dataclass
- Update `scoring_func()` to incorporate new heuristic
- Retrain with GA to find optimal weight

### Training for Specific Metrics

The `--fitness` flag determines optimization target:
- `--fitness lines`: Maximize lines cleared (survival)
- `--fitness score`: Maximize game score (includes level multiplier)

Both use `top_3rd_avg_of()` which runs multiple games and averages top 33% results for robust fitness estimation.

### Debugging Bot Decisions

Enable detailed logging with `--debug` and `--stats`:
```bash
python run.py --bot-model WeightedBotLines --stats --debug
```

This shows:
- Move selection time
- Number of moves considered
- Best move parameters
- Board state comparisons (if replanning occurs)
