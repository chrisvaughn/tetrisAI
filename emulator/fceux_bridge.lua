-- FCEUX Lua bridge for TetrisAI bot
-- Reads commands from CMD_FILE each frame and drives joypad state.
--
-- Command format (written to CMD_FILE by Python):
--   press:<buttons>:<hold_frames>:<gap_frames>
--   press_seq:<step>|<step>|...:<hold_frames>:<gap_frames>
--   hold:<button>
--   release:<button>
--
-- Buttons are comma-separated NES names: A, B, start, select, up, down, left, right
-- For press_seq, each <step> is itself a comma-separated set of buttons to
-- press simultaneously; steps are executed back-to-back (hold+gap each) with
-- a single ack after the final step's gap completes. This lets Python send a
-- whole piece's keypress sequence as one blocking IPC round trip instead of
-- one per keypress.
-- After each command is consumed, acks by creating ACK_FILE.
--
-- State machine per frame (via emu.registerbefore):
--   idle     → read CMD_FILE → pressing
--   pressing → count down hold_frames → waiting
--   waiting  → count down gap_frames → idle (write ack), or advance to the
--              next queued step (press_seq) without acking
--   hold/release commands are processed in idle state with immediate ack

local CMD_FILE = "/tmp/fceux_tetris_cmd"
local ACK_FILE = "/tmp/fceux_tetris_ack"

local state = "idle"
local press_keys = {}
local hold_keys = {}
local press_frames_left = 0
local gap_frames_left = 0

-- press_seq queue: list of button-sets (tables), executed one per
-- pressing/waiting cycle using seq_hold_frames/seq_gap_frames.
local press_queue = {}
local press_queue_idx = 0
local seq_hold_frames = 0
local seq_gap_frames = 0

local function write_ack()
    local f = io.open(ACK_FILE, "w")
    if f then
        f:write("ok\n")
        f:close()
    end
end

local function parse_buttons(s)
    local t = {}
    for btn in s:gmatch("[^,]+") do
        t[btn] = true
    end
    return t
end

local function build_buttons()
    local buttons = {}
    for k, _ in pairs(hold_keys) do
        buttons[k] = true
    end
    if state == "pressing" then
        for k, _ in pairs(press_keys) do
            buttons[k] = true
        end
    end
    return buttons
end

local function read_command()
    local f = io.open(CMD_FILE, "r")
    if not f then return nil end
    local line = f:read("*l")
    f:close()
    os.remove(CMD_FILE)
    return line
end

local function process_command(line)
    -- split on ':'
    local parts = {}
    for p in line:gmatch("[^:]+") do
        parts[#parts + 1] = p
    end
    local cmd = parts[1]

    if cmd == "ping" then
        write_ack()

    elseif cmd == "press" then
        press_keys = parse_buttons(parts[2] or "")
        press_frames_left = tonumber(parts[3]) or 2
        gap_frames_left = tonumber(parts[4]) or 1
        state = "pressing"
        -- ack comes after gap completes

    elseif cmd == "press_seq" then
        press_queue = {}
        for step in (parts[2] or ""):gmatch("[^|]+") do
            press_queue[#press_queue + 1] = parse_buttons(step)
        end
        seq_hold_frames = tonumber(parts[3]) or 2
        seq_gap_frames = tonumber(parts[4]) or 1
        if #press_queue == 0 then
            write_ack()
        else
            press_queue_idx = 1
            press_keys = press_queue[press_queue_idx]
            press_frames_left = seq_hold_frames
            gap_frames_left = seq_gap_frames
            state = "pressing"
            -- ack comes after the final step's gap completes
        end

    elseif cmd == "hold" then
        if parts[2] then
            hold_keys[parts[2]] = true
        end
        write_ack()

    elseif cmd == "release" then
        if parts[2] then
            hold_keys[parts[2]] = nil
        end
        write_ack()
    end
end

local function on_frame_inner()
    if state == "pressing" then
        press_frames_left = press_frames_left - 1
        if press_frames_left <= 0 then
            press_keys = {}
            state = "waiting"
        end

    elseif state == "waiting" then
        gap_frames_left = gap_frames_left - 1
        if gap_frames_left <= 0 then
            if press_queue_idx > 0 and press_queue_idx < #press_queue then
                press_queue_idx = press_queue_idx + 1
                press_keys = press_queue[press_queue_idx]
                press_frames_left = seq_hold_frames
                gap_frames_left = seq_gap_frames
                state = "pressing"
            else
                press_queue = {}
                press_queue_idx = 0
                state = "idle"
                write_ack()
            end
        end

    elseif state == "idle" then
        local line = read_command()
        if line then
            process_command(line)
        end
    end

    joypad.set(1, build_buttons())
end

local function on_frame()
    local ok, err = pcall(on_frame_inner)
    if not ok then
        print("TetrisAI bridge error: " .. tostring(err))
        -- reset to idle so we can keep going
        state = "idle"
        press_keys = {}
        press_queue = {}
        press_queue_idx = 0
        write_ack()
    end
end

emu.registerbefore(on_frame)
print("TetrisAI FCEUX bridge loaded. Waiting for commands on " .. CMD_FILE)
