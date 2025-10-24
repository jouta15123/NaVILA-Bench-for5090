from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import gzip
import json
import numpy as np
import textwrap


@dataclass
class InstructionData:
    """Container for natural language navigation instructions."""

    instruction_text: str
    instruction_tokens: Optional[List[str]] = field(default=None)


def skip(*args, **kwargs):
    """Compatibility stub for deprecated evaluator callbacks."""


def read_episodes(file_path: str):
    """Read Matterport navigation episodes from a gzipped JSON file."""
    with gzip.open(file_path, "rt", encoding="utf-8") as file:
        data = json.load(file)
    return data["episodes"]


def add_instruction_on_img(img: np.ndarray, text: str, start_y: int = 0) -> None:
    """Overlay an instruction string onto an RGB image in-place."""
    font_size = 0.6
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    char_size = cv2.getTextSize(" ", font, font_size, thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int((img.shape[1] - 15) / char_size[0]))
    if len(wrapped_text) < 8:
        wrapped_text.insert(0, "")

    y = start_y
    start_x = 15
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, thickness)[0]
        y += textsize[1] + 25
        cv2.putText(
            img,
            line,
            (start_x, y),
            font,
            font_size,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )


def get_vel_command(text: str):
    """Map a textual VLM command to a base velocity and execution duration."""
    lowered = text.lower()
    if "turn left" in lowered:
        if "45" in lowered:
            return [0.0, 0.0, np.pi / 6.0], 1.5
        if "30" in lowered:
            return [0.0, 0.0, np.pi / 6.0], 1.0
        if "15" in lowered:
            return [0.0, 0.0, np.pi / 6.0], 0.5
        return [0.0, 0.0, np.pi / 6.0], 0.5
    if "turn right" in lowered:
        if "45" in lowered:
            return [0.0, 0.0, -np.pi / 6.0], 1.5
        if "30" in lowered:
            return [0.0, 0.0, -np.pi / 6.0], 1.0
        if "15" in lowered:
            return [0.0, 0.0, -np.pi / 6.0], 0.5
        return [0.0, 0.0, -np.pi / 6.0], 0.5
    if "move forward" in lowered or "move" in lowered:
        if "75" in lowered:
            return [0.5, 0.0, 0.0], 1.5
        if "50" in lowered:
            return [0.5, 0.0, 0.0], 1.0
        if "25" in lowered:
            return [0.5, 0.0, 0.0], 0.5
        return [0.5, 0.0, 0.0], 0.5
    if "stop" in lowered:
        return [0.0, 0.0, 0.0], 0.0
    return [0.5, 0.0, 0.0], 0.5
