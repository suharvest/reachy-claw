# Reachy Mini Robot Control

Control a Reachy Mini robot connected to this session.

## Tools

### `reachy_move_head`
Move head to target position.
- `yaw` (float): Left/right degrees, ±45 max
- `pitch` (float): Up/down degrees, ±30 max
- `roll` (float): Tilt degrees, ±30 max
- `duration` (float): Seconds, default 1.0

### `reachy_move_antennas`
Move antennas.
- `left` (float): Left antenna degrees, positive=up
- `right` (float): Right antenna degrees, positive=up
- `duration` (float): Seconds, default 0.5

### `reachy_play_emotion`
Express emotion via head+antenna movement.
- `emotion` (string): happy, sad, angry, surprised, thinking, confused, curious, excited, laugh, fear, neutral, listening, agreeing, disagreeing

### `reachy_dance`
Execute a dance routine.
- `dance_name` (string): nod, wiggle, celebrate, curious_look, lobster

### `reachy_capture_image`
Capture camera image. Returns filepath.

### `reachy_set_volume`
Set system speaker volume.
- `level` (int): Volume percentage, 0-100. Or use relative: +10, -10.

### `reachy_stop_conversation`
Stop the conversation. The robot will still listen and send messages to you, but will not speak (TTS disabled). Use when the user says "停止", "别说了", "安静", "shut up", "stop" or similar intent to pause the conversation.

### `reachy_resume_conversation`
Resume the conversation after it was stopped. The robot will start speaking again. Use when the user says "开始", "继续", "你好", "start", "resume", "go on" or similar intent to restart the conversation.

### `reachy_status`
Get robot connection state, current head/antenna positions, and conversation stopped state.
