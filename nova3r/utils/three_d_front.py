"""Shared parsing and coordinate conventions for 3D-FRONT preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re

import numpy as np


# Blender's glTF importer maps glTF (x, y, z), where +Y is up, to Blender
# (x, -z, y), where +Z is up. Column vectors are used throughout the tools.
GLTF_TO_BLENDER = np.asarray(
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)
BLENDER_CAMERA_TO_OPENCV = np.diag(
    np.asarray([1.0, -1.0, -1.0, 1.0], dtype=np.float32)
)


def world_to_opencv_camera(blender_world_to_camera, face_world_to_camera=None):
    """Compose the canonical 3D-FRONT glTF-world to OpenCV-camera transform.

    ``blender_world_to_camera`` is the matrix saved by the panorama renderer.
    An optional cubemap-face transform maps panorama OpenCV coordinates into
    the selected face's OpenCV coordinates.
    """
    panorama = (
        BLENDER_CAMERA_TO_OPENCV
        @ np.asarray(blender_world_to_camera, dtype=np.float32)
        @ GLTF_TO_BLENDER
    )
    if panorama.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 panorama transform, got {panorama.shape}")
    if face_world_to_camera is None:
        return panorama
    face = np.asarray(face_world_to_camera, dtype=np.float32)
    if face.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 cubemap-face transform, got {face.shape}")
    return face @ panorama

_PANORAMA_NAME = re.compile(r"^(?P<room>.+)_panorama_(?P<frame>\d+)\.[^.]+$")


@dataclass(frozen=True)
class PanoramaSample:
    room: str
    frame: int


def parse_panorama_sample(value: str) -> PanoramaSample:
    """Parse ``scene/room/room_panorama_N.png`` or ``scene/room N``."""
    value = str(value).strip()
    fields = value.split()
    if len(fields) == 2:
        room, frame_text = fields
        if not frame_text.isdigit():
            raise ValueError(f"Invalid 3D-FRONT panorama index in {value!r}")
        room_path = PurePosixPath(room)
        if len(room_path.parts) != 2:
            raise ValueError(f"Expected a scene/room identifier in {value!r}")
        return PanoramaSample(room_path.as_posix(), int(frame_text))
    if len(fields) != 1:
        raise ValueError(f"Invalid 3D-FRONT panorama sample {value!r}")

    path = PurePosixPath(fields[0])
    if len(path.parts) != 3:
        raise ValueError(
            f"Expected scene/room/room_panorama_N.png in {value!r}"
        )
    match = _PANORAMA_NAME.match(path.name)
    if match is None or match.group("room") != path.parent.name:
        raise ValueError(f"Invalid 3D-FRONT panorama filename in {value!r}")
    return PanoramaSample(path.parent.as_posix(), int(match.group("frame")))


def parse_room_id(value: str) -> str:
    """Return ``scene/room`` from a room id or either panorama-list format."""
    value = str(value).strip()
    path = PurePosixPath(value)
    if len(value.split()) == 2 or len(path.parts) == 3:
        return parse_panorama_sample(value).room
    if len(path.parts) != 2:
        raise ValueError(f"Expected scene/room identifier, got {value!r}")
    return path.as_posix()
