-- NES Tetris State Logger for FCEUX
-- Logs one "lock" event per piece — written when $00BF (next-piece ID) changes,
-- which happens during ARE after the current piece locks.  At that moment:
--   $0062 = the piece that just locked (combined type+rotation ID)
--   $0060 = its final column (0-based, leftmost filled cell or bounding-box left)
--   $0061 = its final row    (0-based, topmost filled cell or bounding-box top)
--   board  = post-line-clear state (exactly what the Python engine should produce)
--
-- Usage: File > Load Lua Script while playing.
-- Output: tetris_log.jsonl in FCEUX's working directory.
--
-- Board string: 200 chars, "0"=empty "1"=filled, row-major top-left first.
--
-- $0062 combined piece ID scheme (0–19):
--   T:  0  1  2  3   J:  4  5  6  7   Z:  8  9
--   S: 10 11         O: 12 13         L: 14 15 16 17   I: 18 19
--
-- Python Tetrominoes: 0=i  1=l  2=j  3=o  4=s  5=t  6=z

-- ── RAM addresses (US NTSC PRG0, all confirmed) ─────────────────────────────
local ADDR_PLAY_STATE  = 0x0048  -- 10=playing, 5=line-clear, 6=ARE
local ADDR_PIECE_X     = 0x0060  -- current piece column (0-based pivot)
local ADDR_PIECE_Y     = 0x0061  -- current piece row    (0-based pivot)
-- $0062 echoes $0042 for most pieces but becomes 0x13 (sentinel) for I-piece horizontal.
-- Log both so compare_log.py can pick the authoritative one.
local ADDR_ORIENT_A    = 0x0042  -- orientation ID per meatfighter (0–18; 0x13=sentinel)
local ADDR_ORIENT_B    = 0x0062  -- mirrors $0042 except for I-piece (use $0042 if they differ)
local ADDR_LEVEL       = 0x0064  -- level (raw byte, not BCD)
local ADDR_NEXT_PIECE  = 0x00BF  -- next-piece combined ID (changes during ARE)
local ADDR_PRNG_LO     = 0x0017
local ADDR_PRNG_HI     = 0x0018
local ADDR_BOARD       = 0x0400  -- 200 bytes, 20×10, row-major

local ADDR_LINES_HI    = 0x0050  -- lines BCD: tens+ones
local ADDR_LINES_LO    = 0x0051  -- lines BCD: hundreds
local ADDR_SCORE_1     = 0x0053  -- score BCD byte 1
local ADDR_SCORE_2     = 0x0054  -- score BCD byte 2
local ADDR_SCORE_3     = 0x0055  -- score BCD byte 3

local EMPTY_TILE       = 0xEF

local STATE_PLAYING    = 10
local STATE_LINE_CLEAR = 5
-- local STATE_ARE     = 6  (not used directly)

-- ── Orientation ID decode tables (meatfighter $8A9C, IDs 0x00–0x12) ─────────
-- T=0-3  J=4-7  Z=8-9  O=10  S=11-12  L=13-16  I=17-18  (0x13=19 is sentinel)
local ID_TO_TYPE = {
    [0]=0,[1]=0,[2]=0,[3]=0,          -- T
    [4]=1,[5]=1,[6]=1,[7]=1,          -- J
    [8]=2,[9]=2,                       -- Z
    [10]=3,                            -- O (1 orientation)
    [11]=4,[12]=4,                     -- S
    [13]=5,[14]=5,[15]=5,[16]=5,       -- L
    [17]=6,[18]=6,[19]=6               -- I (19 = second-vertical sentinel state)
}

local ID_TO_ROT = {
    [0]=0,[1]=1,[2]=2,[3]=3,
    [4]=0,[5]=1,[6]=2,[7]=3,
    [8]=0,[9]=1,
    [10]=0,
    [11]=0,[12]=1,
    [13]=0,[14]=1,[15]=2,[16]=3,
    [17]=0,[18]=1,[19]=0               -- 19 = same vertical orientation as 17
}

-- ── Output ──────────────────────────────────────────────────────────────────
print("[tetris_logger] loading...")

local LOG_FILE = "tetris_log.jsonl"
local f, open_err = io.open(LOG_FILE, "w")
if not f then
    print("[tetris_logger] ERROR: cannot open " .. LOG_FILE .. ": " .. tostring(open_err))
    return
end

-- ── Helpers ─────────────────────────────────────────────────────────────────
local function bcd(byte)
    return math.floor(byte / 16) * 10 + (byte % 16)
end

local function read_lines()
    return bcd(memory.readbyte(ADDR_LINES_LO)) * 100
         + bcd(memory.readbyte(ADDR_LINES_HI))
end

local function read_score()
    return bcd(memory.readbyte(ADDR_SCORE_1)) * 10000
         + bcd(memory.readbyte(ADDR_SCORE_2)) * 100
         + bcd(memory.readbyte(ADDR_SCORE_3))
end

local function read_prng()
    -- $0017 (lo byte) holds the high bits of the 16-bit LFSR value;
    -- $0018 (hi byte) holds the low bits. Swap so the result matches the
    -- polynomial the Python nes_prng() step function operates on.
    return memory.readbyte(ADDR_PRNG_LO) * 256 + memory.readbyte(ADDR_PRNG_HI)
end

local function read_board()
    local t = {}
    for i = 0, 199 do
        t[i + 1] = (memory.readbyte(ADDR_BOARD + i) == EMPTY_TILE) and "0" or "1"
    end
    return table.concat(t)
end

-- ── Diagnostic: log all state+orientation changes ──────────────────────────
-- Prints whenever state or orient_a changes.  Specifically helps us see what
-- $0042 contains during the I-piece horizontal sentinel event (orient_a=19).
-- Also prints $0040/$0041 alongside $0060/$0061 to compare the two position pairs.
local prev_diag_state  = -1
local prev_diag_orient = -1
emu.registerafter(function()
    local state   = memory.readbyte(ADDR_PLAY_STATE)
    local orient  = memory.readbyte(ADDR_ORIENT_A)
    if state ~= prev_diag_state or orient ~= prev_diag_orient then
        print(string.format(
            "s=%d->%d  oid=%d  f=%d  x40=%d y41=%d  x60=%d y60=%d  obf=%d",
            prev_diag_state, state, orient, emu.framecount(),
            memory.readbyte(0x0040), memory.readbyte(0x0041),
            memory.readbyte(ADDR_PIECE_X), memory.readbyte(ADDR_PIECE_Y),
            memory.readbyte(ADDR_NEXT_PIECE)))
        prev_diag_state  = state
        prev_diag_orient = orient
    end
end)

-- ── Main logging callback ───────────────────────────────────────────────────
-- Fires when $00BF changes during gameplay.  At that moment the NES has just
-- advanced the PRNG and chosen a new next-piece.  $0062/$0060/$0061 still hold
-- the just-locked piece's final state, and the board is post-line-clear.
-- This gives us everything needed to replay and verify the Python engine.
--
-- prev_next_id stays -1 until the first gameplay frame so the initial $00BF
-- value (set during menus) doesn't trigger a spurious event.

local piece_index  = 0
local prev_next_id = -1  -- -1 = not yet seen during gameplay

emu.registerafter(function()
    local state   = memory.readbyte(ADDR_PLAY_STATE)
    local next_id = memory.readbyte(ADDR_NEXT_PIECE)

    -- Skip menu states — do NOT update prev_next_id so first gameplay
    -- $00BF change is correctly detected.
    if state < STATE_LINE_CLEAR then
        return
    end

    if next_id ~= prev_next_id and prev_next_id >= 0 then
        piece_index = piece_index + 1
        -- Use $0042 (authoritative orientation ID, 0-18) if it differs from $0062.
        -- $0062 becomes 0x13 (19, sentinel) for I-piece horizontal; $0042 stays correct.
        local orient_a = memory.readbyte(ADDR_ORIENT_A)
        local orient_b = memory.readbyte(ADDR_ORIENT_B)
        local pid = (orient_a ~= 19) and orient_a or orient_b
        f:write(string.format(
            '{"event":"lock","i":%d,"frame":%d,"piece_id":%d,"piece_type":%d,"piece_rot":%d,"orient_a":%d,"orient_b":%d,"next_id":%d,"x":%d,"y":%d,"level":%d,"lines":%d,"score":%d,"prng":%d,"board":"%s"}\n',
            piece_index, emu.framecount(),
            pid,
            ID_TO_TYPE[pid] or -1,
            ID_TO_ROT[pid] or 0,
            orient_a, orient_b,
            next_id,
            memory.readbyte(ADDR_PIECE_X),
            memory.readbyte(ADDR_PIECE_Y),
            memory.readbyte(ADDR_LEVEL),
            read_lines(), read_score(), read_prng(),
            read_board()
        ))
        f:flush()
    end

    prev_next_id = next_id
end)

print(string.format("[tetris_logger] OK — writing to %s. Play a game!", LOG_FILE))
