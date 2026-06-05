# Improving Training

Identified improvements to make the bot train on a simulation that better matches real emulator play.

---

## 1. Emulator/Simulation Mismatch (Implemented)

**Problem:** The in-memory simulation treated moves as instantaneous. In the emulator, each keypress costs ~44ms during which the piece is falling at level speed. At level 19 (2 frames/row ≈ 33ms/row), a 6-keypress move lets the piece fall ~8 rows before it reaches its target column. The bot was therefore trained to make moves it sometimes can't execute in time.

**What was implemented:**
- `execute_move` in `bot/weighted_bot/evaluate.py` pre-drops the piece by the number of rows it would fall during key execution (based on `state.frames_per_cell`). Moves that can't be completed from the dropped position raise `InvalidMove` and are filtered out.
- `frames_per_cell` now updates as the level advances during a game (see item 3), so the fall simulation stays accurate throughout the 19–28 plateau.

**Note:** The 44ms/keypress constant (`MS_PER_KEYPRESS` in `tetris/constants.py`) reflects the current keyboard code's hold+gap timing. If that timing is tuned, this constant should be updated to match.

---

## 2. Move Cost as a Trainable Weight (Implemented)

**Problem:** Even with the fall simulation, there may be value in having the bot explicitly prefer simpler move sequences as a soft bias, rather than just filtering out impossible ones.

**What was implemented:** Added a `move_cost` field to `Weights` in `bot/weighted_bot/evaluate.py`. In `execute_and_score`, after scoring the state:

```python
keypresses = (1 if rot == 3 else rot) + abs(translation)
score += weights.move_cost * keypresses
```

The GA learns the optimal penalty value (expected to be negative — more keypresses = worse). This is complementary to the fall simulation, not a replacement for it.

---

## 3. Level Progression During Training (Implemented)

**Problem:** `self.level` in `game.py` was set once at initialization and never updated. Real NES Tetris has a specific advancement rule: when starting at level 19, the first level-up (19→20) requires **140 lines** cleared (due to a NES counting bug — not 10 or 200), then every subsequent level advances every 10 lines. The full 19–28 plateau therefore spans **230 lines** (140 + 9×10). At level 29, drop speed increases to 1 frame/cell, which the bot can't handle.

**What was implemented:** `_advance_level_if_needed()` in `game.py` uses the authentic NES formula (`min(L*10+10, max(100, L*10-50))`) for the first-advance threshold, then increments the level every 10 lines after that. It updates `self.level`, `self.frames_per_cell`, and `self.state.frames_per_cell` on each advance. Called immediately after `self.lines += lines` so the score for that piece uses the newly advanced multiplier. Combined with the 230-line cap (item 6), the simulation stays within the plateau and never reaches level 29.

---

## 4. `top_3rd_avg_of` Naming Bug (Fixed)

In `train.py`, the fitness function was called `top_3rd_avg_of` but actually averaged the **top 67%** of results (it discarded the bottom 33%):

```python
results = results[int(len(results) * 0.33):]  # keeps top 67%, not top 33%
return sum(results) / len(results)
```

**Fix:** Renamed to `top_two_thirds_avg_of` to match the actual behavior.

---

## 6. Cap Training Games at 230 Lines (Implemented)

**Insight from meatfighter.com/tetrisairevisited:** The article trains on exactly "100 runs of the 19–28 plateau" — each evaluation game stops at 230 lines cleared rather than running to game over. The 230-line cap corresponds to the full plateau: 140 lines for 19→20, then 10 per level through 28→29.

**What was implemented:** Added `--max-lines` flag (default 230) to `train.py`. `evaluate_with_bot` breaks the game loop when `game.lines >= max_lines`. Pass `--max-lines 0` to disable. The cap is threaded through `GenomeFitness` so it applies consistently across all genome evaluations.

---

## 7. Save Piece Lists to Pickle for Consistent Resumption (Implemented)

**Problem:** In `train.py`, `piece_lists` was generated fresh each call to `main()` using random seeds. When training resumed (`resume=True`), `SaveState` only restored genomes and generation count — new random piece lists were created. This meant fitness values were not comparable across resume sessions.

**What was implemented:** Added `piece_lists` to `SaveState` in `evolution.py`. `GA` accepts and stores `piece_lists`, saves them each generation, and restores them on resume. In `train.py::main()`, the save file is read before generating piece lists so that resumed sessions reuse the same sequences.

---

## 8. Convergence Detection and Restart in GA (Implemented)

**Insight from meatfighter.com/tetrisairevisited:** The article's PSO automatically restarts with new random weights when no advancement is detected after a certain number of iterations.

**Problem:** The GA had no escape from premature convergence. Once the top 15 genomes become similar, crossover produces near-copies and progress stalls silently.

**What was implemented:** `_is_stalled()` in `evolution.py` checks whether best fitness has improved by less than 0.5% over the last 20 generations. When a stall is detected, `_restart_from_best()` replaces the population with the current best genome plus `population_size - 1` new genomes generated by perturbing the best's weights with Gaussian noise (σ=0.5). The best genome is always preserved across the restart.

---

## 9. Row Transitions Heuristic (Implemented)

**Insight from meatfighter.com/tetrisairevisited:** The article uses horizontal row transitions (count of filled→empty and empty→filled flips scanning each row) as one of its 17 evaluation factors.

**What was implemented:** Added `count_row_transitions()` to `Board` (vectorized: pads each row with wall cells and counts `diff != 0` transitions), exposed as `GameState.row_transitions()`, added `row_transitions: float = 0` to `Weights`, and included in `scoring_v2`. The GA evolves its weight (expected to be negative — more transitions = more fragmented board).

---

## 10. Spawn Point Protection (Implemented)

**Insight from meatfighter.com/tetrisairevisited:** One of four "pacification rules" — reject placements that would fill cells in the spawn zone (approximately columns 4–6, rows 0–1).

**What was implemented:** Added `Board.spawn_zone_filled()` which counts filled cells in rows 0–1, columns 3–6 (0-indexed, matching the spawn center at x=6 1-based). Both `scoring_v1` and `scoring_v2` apply a hard `−1000 × spawn_blocked` penalty when any spawn zone cells are filled, consistent with the existing `unreachable_cells` penalty pattern.

---

## 11. Greedy Tetris Targeting (Implemented)

**Insight from meatfighter.com/tetrisairevisited:** One of four "pacification rules" — when a Tetris (4-line clear) is available, always prefer it over moves that clear fewer lines, as a hard rule rather than relying on the `lines` weight.

**What was implemented:** In `best_move()`, added `1 if x.lines_completed == 4 else 0` as the highest-priority sort key. Any move clearing 4 lines now sorts above all moves clearing fewer lines, regardless of weighted score.

---

## 12. Training Speed (Ongoing)

Each generation with 100 genomes × 100 piece-list games takes ~6-7 hours of compute with 8 genome-level parallel workers on a 10-core machine. Knobs to reduce wall-clock time:

- **Fewer piece lists:** `--num-iterations` defaults to 100. Reducing to 20-30 gives a noisier fitness signal but is 3-5× faster per generation.
- **Smaller population:** `--population` defaults to 100. Smaller populations converge faster but with less diversity.
- **Fewer generations:** The fitness improved 5× in generation 1 alone. Convergence may be mostly done by generation 20-30.
- **Genome workers:** Currently 8 on a 10-core machine. Already well-utilized.
