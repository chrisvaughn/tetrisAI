# Improving Training

Identified improvements to make the bot train on a simulation that better matches real emulator play.

---

## 1. Emulator/Simulation Mismatch (Partially Implemented)

**Problem:** The in-memory simulation treats moves as instantaneous. In the emulator, each keypress costs ~44ms during which the piece is falling at level speed. At level 19 (2 frames/row ≈ 33ms/row), a 6-keypress move lets the piece fall ~8 rows before it reaches its target column. The bot is therefore trained to make moves it sometimes can't execute in time.

**What was implemented:** `execute_move` in `bot/weighted_bot/evaluate.py` now pre-drops the piece by the number of rows it would fall during key execution (based on `state.frames_per_cell`). Moves that can't be completed from the dropped position raise `InvalidMove` and are filtered out.

**Still to consider:**

- `frames_per_cell` is fixed at the starting level for the entire game. In real play the level increases every 10 lines, making pieces fall faster. The simulation could update `frames_per_cell` as lines are cleared during training.
- The 44ms/keypress constant (`MS_PER_KEYPRESS` in `tetris/constants.py`) reflects the current keyboard code's hold+gap timing. If that timing is tuned, this constant should be updated to match.

---

## 2. Move Cost as a Trainable Weight (Not Implemented)

**Problem:** Even with the fall simulation, there may be value in having the bot explicitly prefer simpler move sequences as a soft bias, rather than just filtering out impossible ones.

**Proposed change:** Add a `move_cost` field to `Weights` in `bot/weighted_bot/evaluate.py` and include it in the scoring functions:

```python
keypresses = (1 if rot == 3 else rot) + abs(translation)
score += weights.move_cost * keypresses
```

The GA would then learn the optimal penalty value (expected to be negative — more keypresses = worse). This is complementary to the fall simulation, not a replacement for it.

---

## 3. Level Progression During Training (Not Implemented)

**Problem:** `self.level` in `game.py` is set once at initialization and never updated. Real NES Tetris has a specific advancement rule: when starting at level 19, the first level-up (19→20) requires **140 lines** cleared (due to a NES counting bug — not 10 or 200), then every subsequent level advances every 10 lines. The full 19–28 plateau therefore spans **230 lines** (140 + 9×10). At level 29, drop speed increases to 1 frame/cell, which the bot can't handle.

Two consequences:

1. **Drop speed never updates** — the simulation always runs at level 19 speed (2 frames/cell). This is accidentally correct for the 19–28 plateau since that speed is constant, but it means the game never ends from hitting level 29 — it runs until the bot dies.
2. **Score multiplier is wrong** — score is calculated as `score_by_number_of_lines_cleared[lines-1] * (self.level + 1)`, which stays at ×20 forever. In real play it increases to ×21, ×22... up to ×29 as levels advance. This affects `--fitness score` training: all lines earn the same multiplier regardless of when they're cleared.

**Options:**

- Implement the NES advancement rule in `game.py`: track cumulative lines, apply the 140-line first-advance rule, then update `self.level` and `self.frames_per_cell` every 10 lines after that.
- Cap training games at 230 lines (see item 6) so the simulation matches the plateau boundary without needing full level advancement.

---

## 4. `top_3rd_avg_of` Naming Bug (Not Fixed)

In `train.py`, the fitness function is called `top_3rd_avg_of` but actually averages the **top 67%** of results (it discards the bottom 33%):

```python
results = results[int(len(results) * 0.33):]  # keeps top 67%, not top 33%
return sum(results) / len(results)
```

This inflates fitness scores relative to the name's implication. The behavior may be intentional (averaging over a majority gives a more stable signal), but the name is misleading. Consider renaming to `top_two_thirds_avg_of` or changing the slice to keep only the top third.

---

## 6. Cap Training Games at 230 Lines (Not Implemented)

**Insight from meatfighter.com/tetrisairevisited:** The article trains on exactly "100 runs of the 19–28 plateau" — each evaluation game stops at 230 lines cleared rather than running to game over. The 230-line cap corresponds to the full plateau: 140 lines for 19→20, then 10 per level through 28→29.

**Benefits:**

- Makes each evaluation faster for well-performing bots (no time wasted simulating level 29+ where the bot quickly dies anyway)
- Keeps the fitness signal entirely within the relevant speed regime
- Allows more game iterations in the same wall time

**Implementation:** Add a `max_lines` parameter to `evaluate_with_bot` in `train.py` and break the game loop when `game.lines >= max_lines`. The piece list length of 1000 is likely sufficient since 230 lines takes roughly 300–600 piece placements depending on efficiency.

---

## 7. Save Piece Lists to Pickle for Consistent Resumption (Not Implemented)

**Problem:** In `train.py`, `piece_lists` is generated fresh each call to `main()` using random seeds. When training resumes (`resume=True`), `SaveState` only restores genomes and generation count — new random piece lists are created. This means fitness values are not comparable across resume sessions, and the optimizer may spend generations re-adapting to a new piece distribution rather than making genuine progress.

**Fix:** Add `piece_lists` to `SaveState` in `evolution.py` and restore them on resume. The article uses pre-generated, fixed piece sequences throughout training specifically to ensure fair candidate comparisons.

---

## 8. Convergence Detection and Restart in GA (Not Implemented)

**Insight from meatfighter.com/tetrisairevisited:** The article's PSO automatically restarts with new random weights when no advancement is detected after a certain number of iterations.

**Problem:** The current GA has no escape from premature convergence. Once the top 15 genomes become similar, crossover produces near-copies and progress stalls silently.

**Proposed change:** Track best fitness over a rolling window (e.g., last 20 generations). If improvement is below a small epsilon (e.g., 0.5%), restart the population — either fully random or seeded from the current best genome with large mutations to re-introduce diversity.

---

## 9. Row Transitions Heuristic (Not Implemented)

**Insight from meatfighter.com/tetrisairevisited:** The article uses horizontal row transitions (count of filled→empty and empty→filled flips scanning each row) as one of its 17 evaluation factors.

**Gap:** The current `roughness` metric measures vertical column height differences. Row transitions capture a complementary signal — how fragmented the board looks horizontally. A board with many row transitions has irregular cavities that are hard to fill even when column heights look acceptable.

**Implementation:** Add `count_row_transitions()` to `Board`, expose on `GameState`, add `row_transitions: float = 0` to `Weights`, include in `scoring_v2`.

---

## 10. Spawn Point Protection (Not Implemented)

**Insight from meatfighter.com/tetrisairevisited:** One of four "pacification rules" — reject placements that would fill cells in the spawn zone (approximately columns 4–6, rows 0–1).

**Problem:** The bot can play into a state where new pieces cannot spawn, causing an immediate game-over that could have been avoided.

**Implementation:** Apply a hard penalty in `scoring_v1` and `scoring_v2` (similar to the `unreachable_cells` penalty) when the post-placement board has any filled cells in the spawn zone.

---

## 11. Greedy Tetris Targeting (Not Implemented)

**Insight from meatfighter.com/tetrisairevisited:** One of four "pacification rules" — when a Tetris (4-line clear) is available, always prefer it over moves that clear fewer lines, as a hard rule rather than relying on the `lines` weight.

**Problem:** The current `best_move()` sort uses `lines_completed` as a tiebreaker after score. A Tetris will only be chosen if the score also favors it. With poorly tuned weights early in training, the bot may not exploit Tetris opportunities.

**Implementation:** In `best_move()`, add a first-level sort key that gives highest priority to any move clearing 4 lines, regardless of overall score. Or implement as a hard override: if any candidate move clears 4 lines, return the best-scoring one of those directly.

---

## 12. Training Speed (Ongoing)

Each generation with 100 genomes × 100 piece-list games takes ~6-7 hours of compute with 8 genome-level parallel workers on a 10-core machine. Knobs to reduce wall-clock time:

- **Fewer piece lists:** `--num-iterations` defaults to 100. Reducing to 20-30 gives a noisier fitness signal but is 3-5× faster per generation.
- **Smaller population:** `--population` defaults to 100. Smaller populations converge faster but with less diversity.
- **Fewer generations:** The fitness improved 5× in generation 1 alone. Convergence may be mostly done by generation 20-30.
- **Genome workers:** Currently 8 on a 10-core machine. Already well-utilized.
