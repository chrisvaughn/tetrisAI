-- FCEUX Lua bridge for TetrisAI bot
-- Reads commands from CMD_FILE each frame and drives joypad state.
--
-- Command format (written to CMD_FILE by Python):
--   press:<buttons>:<hold_frames>:<gap_frames>
--   hold:<button>
--   release:<button>
--
-- Buttons are comma-separated NES names: A, B, start, select, up, down, left, right
-- After each command is consumed, acks by creating ACK_FILE.
--
-- State machine per frame (via emu.registerbefore):
--   idle     → read CMD_FILE → pressing
--   pressing → count down hold_frames → waiting
--   waiting  → count down gap_frames → idle (write ack)
--   hold/release commands are processed in idle state with immediate ack

local CMD_FILE = "/tmp/fceux_tetris_cmd"
local ACK_FILE = "/tmp/fceux_tetris_ack"

local state = "idle"
local press_keys = {}
local hold_keys = {}
local press_frames_left = 0
local gap_frames_left = 0

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
            state = "idle"
            write_ack()
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
        write_ack()
    end
end

emu.registerbefore(on_frame)
print("TetrisAI FCEUX bridge loaded. Waiting for commands on " .. CMD_FILE)
