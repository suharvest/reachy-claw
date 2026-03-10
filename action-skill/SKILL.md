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

### `reachy_status`
Get robot connection state and current head/antenna positions.
