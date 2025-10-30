"""
Utilities that mirror the richer NaVILA Habitat evaluation pipeline.

The helpers in this module let us reuse the official evaluation logic in the
IsaacSim runners without duplicating large chunks of code.  They cover three
areas:

- History frame sampling / padding before sending frames to the VLM.
- Prompt construction and base64 payload preparation for the socket server.
- Parsing mid-level textual actions into quantised step queues.
"""

from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
import math
from datetime import datetime
from typing import Any, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# Default quantisation used in the paper and Habitat evaluation.
FORWARD_CHOICES_CM: Tuple[int, ...] = (25, 50, 75)
TURN_CHOICES_DEG: Tuple[int, ...] = (15, 30, 45)

# Action ids used by Habitat waypoint envs (stop, forward, turn left, turn right).
DEFAULT_ACTION_CODES = {
    "stop": 0,
    "move_forward": 1,
    "turn_left": 2,
    "turn_right": 3,
}


@dataclass
class ActionCommand:
    """Represents a mid-level navigation command emitted by the VLM."""

    kind: Literal["move", "turn_left", "turn_right", "stop"]
    magnitude: float | None = None
    unit: Literal["cm", "deg", None] = None

    def quantise(
        self,
        distance_choices: Sequence[int] = FORWARD_CHOICES_CM,
        angle_choices: Sequence[int] = TURN_CHOICES_DEG,
    ) -> "ActionCommand":
        """Snap the magnitude to the closest supported value."""
        if self.kind == "move" and self.magnitude is not None:
            closest = min(distance_choices, key=lambda x: abs(x - self.magnitude))
            return ActionCommand(kind="move", magnitude=float(closest), unit="cm")
        if self.kind in {"turn_left", "turn_right"} and self.magnitude is not None:
            closest = min(angle_choices, key=lambda x: abs(x - self.magnitude))
            return ActionCommand(kind=self.kind, magnitude=float(closest), unit="deg")
        return self


def sample_and_pad_images(
    frames: Sequence[Image.Image],
    num_frames: int,
    frame_size: Tuple[int, int] | None = None,
) -> List[Image.Image]:
    """
    Mirror Habitat's sampling: keep evenly spaced history plus the latest frame.

    Args:
        frames: Ordered list of PIL images (oldest -> newest).
        num_frames: Desired number of frames for the VLM.
        frame_size: Optional (width, height) for black padding frames.
    """
    if not frames:
        if frame_size is None:
            frame_size = (512, 512)
        return [Image.new("RGB", frame_size, color=(0, 0, 0)) for _ in range(num_frames)]

    padded = list(frames)
    if frame_size is None:
        frame_size = padded[-1].size

    while len(padded) < num_frames:
        padded.insert(0, Image.new("RGB", frame_size, color=(0, 0, 0)))

    latest = padded[-1]
    if len(padded) == num_frames:
        sampled = padded
    else:
        indices = np.linspace(0, len(padded) - 1, num=num_frames - 1, endpoint=False, dtype=int)
        sampled = [padded[i] for i in indices] + [latest]
    return sampled


def encode_images_to_base64(frames: Sequence[Image.Image]) -> List[str]:
    """JPEG-encode frames and return base64 strings for socket transport."""
    encoded: List[str] = []
    for frame in frames:
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG")
        encoded.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return encoded


def build_vlm_prompt(instruction: str, num_history_frames: int) -> str:
    """
    Reproduce the official prompt template used in the Habitat evaluation.
    """
    interleaved = "<image>\n" * num_history_frames
    return (
        "Imagine you are a robot programmed for navigation tasks. "
        f"You have been given a video of historical observations {interleaved}"
        "and current observation <image>\n. "
        f'Your assigned task is: "{instruction}" '
        "Analyze this series of images to decide your next action, which could be "
        "turning left or right by a specific degree, moving forward a certain distance, "
        "or stop if the task is completed."
    )


_MOVE_PATTERN = re.compile(
    r"(?:move|go|walk|proceed)\s+forward\s+(?:around\s+)?(\d+(?:\.\d+)?)\s*cm",
    re.IGNORECASE,
)
_TURN_LEFT_PATTERN = re.compile(
    r"(?:turn|rotate)\s+left\s+(?:about\s+|approximately\s+)?(\d+(?:\.\d+)?)\s*deg(?:ree[s]?)?",
    re.IGNORECASE,
)
_TURN_RIGHT_PATTERN = re.compile(
    r"(?:turn|rotate)\s+right\s+(?:about\s+|approximately\s+)?(\d+(?:\.\d+)?)\s*deg(?:ree[s]?)?",
    re.IGNORECASE,
)
_STOP_PATTERN = re.compile(r"\bstop\b", re.IGNORECASE)

_INSTRUCTION_MOVE_PATTERN = re.compile(
    r"(?:move|go|walk|proceed|head|continue)\s+(?:straight\s+)?(?:forward\s+)?(?:for\s+|about\s+|approximately\s+)?"
    r"(\d+(?:\.\d+)?)?\s*(cm|centimeter(?:s)?|m|meter(?:s)?|metre(?:s)?)?",
    re.IGNORECASE,
)
_INSTRUCTION_TURN_LEFT_PATTERN = re.compile(
    r"(?:turn|rotate|pivot|swing)\s+left(?:\s+(?:for|about|approximately)\s+(\d+(?:\.\d+)?))?\s*(deg(?:ree)?(?:s)?)?",
    re.IGNORECASE,
)
_INSTRUCTION_TURN_RIGHT_PATTERN = re.compile(
    r"(?:turn|rotate|pivot|swing)\s+right(?:\s+(?:for|about|approximately)\s+(\d+(?:\.\d+)?))?\s*(deg(?:ree)?(?:s)?)?",
    re.IGNORECASE,
)


def _convert_linear_unit(value: float | None, unit: str | None) -> float | None:
    if value is None:
        return None
    if not unit:
        return float(value) * 100.0  # assume metres if unspecified
    unit = unit.lower()
    if unit.startswith("cm"):
        return float(value)
    if unit.startswith(("m", "met")):
        return float(value) * 100.0
    return float(value) * 100.0


def _convert_angular_unit(value: float | None, unit: str | None) -> float | None:
    if value is None:
        return None
    return float(value)


def parse_vlm_response(text: str) -> List[ActionCommand]:
    """
    Parse the VLM text into ordered ActionCommand objects.

    The parser searches for explicit action spans in the order they appear.
    """
    commands: List[ActionCommand] = []
    cursor = 0
    lower_text = text.lower()

    while cursor < len(text):
        match_spans = []
        for kind, pattern in (
            ("move", _MOVE_PATTERN),
            ("turn_left", _TURN_LEFT_PATTERN),
            ("turn_right", _TURN_RIGHT_PATTERN),
        ):
            match = pattern.search(text, cursor)
            if match:
                match_spans.append((match.start(), match.end(), kind, match))
        stop_match = _STOP_PATTERN.search(text, cursor)
        if stop_match:
            match_spans.append((stop_match.start(), stop_match.end(), "stop", stop_match))

        if not match_spans:
            break

        start, end, kind, match_obj = min(match_spans, key=lambda item: item[0])
        cursor = end

        if kind == "move":
            magnitude = float(match_obj.group(1))
            commands.append(ActionCommand(kind="move", magnitude=magnitude, unit="cm"))
        elif kind == "turn_left":
            magnitude = float(match_obj.group(1))
            commands.append(ActionCommand(kind="turn_left", magnitude=magnitude, unit="deg"))
        elif kind == "turn_right":
            magnitude = float(match_obj.group(1))
            commands.append(ActionCommand(kind="turn_right", magnitude=magnitude, unit="deg"))
        else:
            commands.append(ActionCommand(kind="stop"))

    # If nothing matched but we saw a stop token anywhere, issue stop.
    if not commands and _STOP_PATTERN.search(lower_text):
        commands.append(ActionCommand(kind="stop"))

    return commands


def quantise_commands(commands: Iterable[ActionCommand]) -> List[ActionCommand]:
    """Apply standard quantisation to every command."""
    return [cmd.quantise() for cmd in commands]


def parse_instruction_to_commands(text: str) -> List[ActionCommand]:
    """
    Extract intended commands from the raw human instruction.
    """
    commands: List[ActionCommand] = []
    cursor = 0
    lower_text = text.lower()

    while cursor < len(text):
        match_spans = []
        move_match = _INSTRUCTION_MOVE_PATTERN.search(text, cursor)
        if move_match:
            match_spans.append((move_match.start(), move_match.end(), "move", move_match))

        left_match = _INSTRUCTION_TURN_LEFT_PATTERN.search(text, cursor)
        if left_match:
            match_spans.append((left_match.start(), left_match.end(), "turn_left", left_match))

        right_match = _INSTRUCTION_TURN_RIGHT_PATTERN.search(text, cursor)
        if right_match:
            match_spans.append((right_match.start(), right_match.end(), "turn_right", right_match))

        stop_match = _STOP_PATTERN.search(text, cursor)
        if stop_match:
            match_spans.append((stop_match.start(), stop_match.end(), "stop", stop_match))

        if not match_spans:
            break

        start, end, kind, match_obj = min(match_spans, key=lambda item: item[0])
        cursor = end

        if kind == "move":
            value_str, unit = match_obj.group(1), match_obj.group(2)
            magnitude = _convert_linear_unit(float(value_str), unit) if value_str else None
            commands.append(ActionCommand(kind="move", magnitude=magnitude, unit="cm"))
        elif kind == "turn_left":
            value_str, unit = match_obj.group(1), match_obj.group(2)
            magnitude = _convert_angular_unit(float(value_str), unit) if value_str else None
            commands.append(ActionCommand(kind="turn_left", magnitude=magnitude, unit="deg"))
        elif kind == "turn_right":
            value_str, unit = match_obj.group(1), match_obj.group(2)
            magnitude = _convert_angular_unit(float(value_str), unit) if value_str else None
            commands.append(ActionCommand(kind="turn_right", magnitude=magnitude, unit="deg"))
        else:
            commands.append(ActionCommand(kind="stop"))

    if not commands and _STOP_PATTERN.search(lower_text):
        commands.append(ActionCommand(kind="stop"))

    filtered: List[ActionCommand] = []
    for cmd in commands:
        if cmd.kind == "move" and cmd.magnitude is None:
            filtered.append(ActionCommand(kind="move", magnitude=50.0, unit="cm"))
        elif cmd.kind in {"turn_left", "turn_right"} and cmd.magnitude is None:
            filtered.append(ActionCommand(kind=cmd.kind, magnitude=30.0, unit="deg"))
        else:
            filtered.append(cmd)
    return filtered


def expand_command_to_action_ids(
    command: ActionCommand,
    action_codes: dict[str, int] = DEFAULT_ACTION_CODES,
) -> List[int]:
    """
    Expand a quantised command into a queue of discrete action ids.

    Returns:
        List of action ids to be sent sequentially to the environment.
    """
    if command.kind == "stop":
        return [action_codes["stop"]]

    if command.kind == "move":
        magnitude = int(command.magnitude or FORWARD_CHOICES_CM[0])
        steps = max(1, magnitude // FORWARD_CHOICES_CM[0])
        return [action_codes["move_forward"]] * steps

    if command.kind in {"turn_left", "turn_right"}:
        magnitude = int(command.magnitude or TURN_CHOICES_DEG[0])
        steps = max(1, magnitude // TURN_CHOICES_DEG[0])
        key = "turn_left" if command.kind == "turn_left" else "turn_right"
        return [action_codes[key]] * steps

    raise ValueError(f"Unsupported command kind: {command.kind}")


def expand_commands_to_queue(
    commands: Iterable[ActionCommand],
    action_codes: dict[str, int] = DEFAULT_ACTION_CODES,
) -> List[int]:
    """Convert a sequence of ActionCommand into a flat queue of action ids."""
    queue: List[int] = []
    for command in commands:
        queue.extend(expand_command_to_action_ids(command, action_codes=action_codes))
    return queue


def commands_cover_expected(expected: Sequence[ActionCommand], actual: Sequence[ActionCommand]) -> bool:
    """Check whether actual commands cover the expected sequence (kind-wise, in order)."""
    if not expected:
        return True
    expected_kinds = [cmd.kind for cmd in expected]
    actual_kinds = [cmd.kind for cmd in actual]
    idx = 0
    for kind in actual_kinds:
        if idx < len(expected_kinds) and kind == expected_kinds[idx]:
            idx += 1
    return idx == len(expected_kinds)


def commands_to_dicts(commands: Sequence[ActionCommand]) -> List[dict]:
    """Serialize ActionCommand objects for logging."""
    return [
        {
            "kind": command.kind,
            "magnitude": command.magnitude,
            "unit": command.unit,
        }
        for command in commands
    ]


def summarize_commands(commands: Sequence[ActionCommand]) -> str:
    parts: List[str] = []
    for cmd in commands:
        if cmd.magnitude is None:
            parts.append(cmd.kind)
        else:
            value = round(cmd.magnitude, 1)
            if cmd.unit == "cm" and abs(value - round(value)) < 1e-3:
                value = int(round(value))
            unit = cmd.unit or ""
            parts.append(f"{cmd.kind}({value}{unit})")
    return ", ".join(parts) if parts else "<no-command>"


def save_map_image(
    base_dir: str,
    cam_frame: Optional[np.ndarray],
    map_frame: Optional[np.ndarray],
    tag: str,
    subdir: str = "maps",
) -> None:
    if cam_frame is None or map_frame is None:
        return
    snapshot_dir = os.path.join(base_dir, subdir)
    os.makedirs(snapshot_dir, exist_ok=True)
    try:
        combined = np.concatenate([cam_frame, map_frame], axis=1)
        Image.fromarray(combined.astype(np.uint8)).save(
            os.path.join(
                snapshot_dir,
                f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}_{tag}.png",
            )
        )
    except Exception as exc:
        print(f"[WARN] Failed to save map snapshot: {exc}")


def extract_distance_to_goal(measurements: Mapping[str, Any] | None) -> float | None:
    """Extract distance-to-goal value from measurements dict."""
    if not measurements:
        return None
    for key in ("DistanceToGoal", "distance_to_goal", "dist_to_goal"):
        if key in measurements:
            value = measurements[key]
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def commands_to_velocity_plan(
    commands: Iterable[ActionCommand],
    linear_speed_mps: float = 0.5,
    angular_speed_rps: float = math.pi / 6.0,
) -> List[Tuple[np.ndarray, float]]:
    """
    Convert commands to constant-velocity segments.

    Returns:
        List of tuples (velocity_vector, duration_seconds).
    """
    plan: List[Tuple[np.ndarray, float]] = []
    for command in commands:
        if command.kind == "stop":
            plan.append((np.zeros(3, dtype=np.float32), 0.0))
            continue
        if command.kind == "move":
            distance_m = (command.magnitude or FORWARD_CHOICES_CM[0]) / 100.0
            duration = distance_m / max(linear_speed_mps, 1e-6)
            plan.append((np.array([linear_speed_mps, 0.0, 0.0], dtype=np.float32), duration))
            continue
        if command.kind == "turn_left":
            angle_rad = math.radians(command.magnitude or TURN_CHOICES_DEG[0])
            duration = angle_rad / max(angular_speed_rps, 1e-6)
            plan.append((np.array([0.0, 0.0, angular_speed_rps], dtype=np.float32), duration))
            continue
        if command.kind == "turn_right":
            angle_rad = math.radians(command.magnitude or TURN_CHOICES_DEG[0])
            duration = abs(angle_rad) / max(angular_speed_rps, 1e-6)
            plan.append((np.array([0.0, 0.0, -angular_speed_rps], dtype=np.float32), duration))
            continue
        raise ValueError(f"Unsupported command kind: {command.kind}")
    return plan
