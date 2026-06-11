# NES Tetris game constants

MS_PER_FRAME = 1000.0 / 60.0  # ~16.667ms per NTSC frame

# Frames the piece takes to fall one row, by level (authentic NES Tetris values)
frames_per_cell_by_level = {
    0: 48,
    1: 43,
    2: 38,
    3: 33,
    4: 28,
    5: 23,
    6: 18,
    7: 13,
    8: 8,
    9: 6,
    10: 5,
    11: 5,
    12: 5,
    13: 4,
    14: 4,
    15: 4,
    16: 3,
    17: 3,
    18: 3,
    19: 2,
    20: 2,
    21: 2,
    22: 2,
    23: 2,
    24: 2,
    25: 2,
    26: 2,
    27: 2,
    28: 2,
    29: 1,
}

# Time cost per keypress for the macOS keyboard→emulator path (Nestopia):
# 20ms hold + 24ms min gap between presses = 44ms per move
MS_PER_KEYPRESS = 44.0

# Time cost per keypress for the FCEUX file-based IPC path:
# Lua bridge takes 2 NES frames per press (hold + gap) = 2 × 16.67ms ≈ 33ms
MS_PER_KEYPRESS_FCEUX = 33.0
