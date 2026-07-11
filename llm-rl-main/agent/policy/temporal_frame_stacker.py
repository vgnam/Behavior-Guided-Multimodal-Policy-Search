"""Online temporal superposition for sampled visual policy trajectories."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image


class TemporalFrameStacker:
    """Collapse every rendered frame into one time-coloured motion-trail image.

    Supplied frames are processed online, so a long MuJoCo rollout does not
    retain hundreds of full-resolution arrays. A slowly updated background model keeps
    static scenery sharp; pixels that differ from that background accumulate a
    motion trail.  Trail colour progresses blue -> cyan -> yellow -> red.
    """

    def __init__(
        self,
        motion_threshold: float = 18.0,
        background_learning_rate: float = 0.01,
        tint_strength: float = 0.45,
        occupancy_scale: float = 4.0,
        max_side: Optional[int] = 1024,
    ):
        if motion_threshold < 0:
            raise ValueError("motion_threshold must be non-negative")
        if not 0.0 <= background_learning_rate <= 1.0:
            raise ValueError("background_learning_rate must be in [0, 1]")
        if not 0.0 <= tint_strength <= 1.0:
            raise ValueError("tint_strength must be in [0, 1]")
        if occupancy_scale <= 0:
            raise ValueError("occupancy_scale must be positive")

        self.motion_threshold = float(motion_threshold)
        self.background_learning_rate = float(background_learning_rate)
        self.tint_strength = float(tint_strength)
        self.occupancy_scale = float(occupancy_scale)
        self.max_side = max_side

        self.frame_count = 0
        self._background: Optional[np.ndarray] = None
        self._rgb_sum: Optional[np.ndarray] = None
        self._time_sum: Optional[np.ndarray] = None
        self._occupancy: Optional[np.ndarray] = None
        self._last_frame: Optional[np.ndarray] = None

    @staticmethod
    def _to_rgb_uint8(frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("Frames must have shape (H, W, 3) or (H, W, 4)")
        array = array[..., :3]
        if array.dtype != np.uint8:
            if array.size and float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        return array

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        if not self.max_side or max(frame.shape[:2]) <= self.max_side:
            return frame
        scale = self.max_side / max(frame.shape[:2])
        width = max(1, int(round(frame.shape[1] * scale)))
        height = max(1, int(round(frame.shape[0] * scale)))
        return np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))

    def add(self, frame: np.ndarray) -> None:
        """Add one frame. Every call contributes to the final image."""
        rgb = self._resize_if_needed(self._to_rgb_uint8(frame)).astype(np.float32)
        if self._background is None:
            height, width = rgb.shape[:2]
            self._background = rgb.copy()
            self._rgb_sum = np.zeros((height, width, 3), dtype=np.float32)
            self._time_sum = np.zeros((height, width), dtype=np.float32)
            self._occupancy = np.zeros((height, width), dtype=np.float32)
        elif rgb.shape != self._background.shape:
            target_size = (self._background.shape[1], self._background.shape[0])
            rgb = np.asarray(
                Image.fromarray(rgb.astype(np.uint8)).resize(
                    target_size, Image.Resampling.LANCZOS
                )
            ).astype(np.float32)

        difference = np.mean(np.abs(rgb - self._background), axis=2)
        motion_mask = difference >= self.motion_threshold

        # If almost the whole image changes (e.g. tracking camera), use a soft
        # difference weight rather than classifying the complete canvas as motion.
        if float(np.mean(motion_mask)) > 0.85:
            soft_mask = np.clip(difference / max(self.motion_threshold * 4.0, 1.0), 0.0, 1.0)
        else:
            soft_mask = motion_mask.astype(np.float32)

        self._rgb_sum += rgb * soft_mask[..., None]
        self._time_sum += float(self.frame_count) * soft_mask
        self._occupancy += soft_mask

        stable_weight = (1.0 - soft_mask)[..., None] * self.background_learning_rate
        self._background = (1.0 - stable_weight) * self._background + stable_weight * rgb
        self._last_frame = rgb
        self.frame_count += 1

    @staticmethod
    def _time_colour(normalized_time: np.ndarray) -> np.ndarray:
        """Map [0, 1] time to blue -> cyan -> yellow -> red RGB."""
        stops = np.array(
            [[35.0, 90.0, 255.0], [35.0, 220.0, 220.0], [245.0, 220.0, 40.0], [240.0, 55.0, 45.0]],
            dtype=np.float32,
        )
        scaled = np.clip(normalized_time, 0.0, 1.0) * (len(stops) - 1)
        lower = np.floor(scaled).astype(int)
        upper = np.minimum(lower + 1, len(stops) - 1)
        fraction = (scaled - lower)[..., None]
        return stops[lower] * (1.0 - fraction) + stops[upper] * fraction

    def finalize(self) -> Optional[np.ndarray]:
        """Return one RGB uint8 superposition image, or None if no frame arrived."""
        if self.frame_count == 0 or self._background is None:
            return None

        occupancy = self._occupancy
        active = occupancy > 1e-6
        safe_occupancy = np.maximum(occupancy, 1e-6)
        foreground = self._rgb_sum / safe_occupancy[..., None]
        mean_time = self._time_sum / safe_occupancy
        if self.frame_count > 1:
            mean_time /= self.frame_count - 1
        else:
            mean_time.fill(0.0)

        time_colour = self._time_colour(mean_time)
        coloured_foreground = (
            (1.0 - self.tint_strength) * foreground
            + self.tint_strength * time_colour
        )
        alpha = 1.0 - np.exp(-occupancy / self.occupancy_scale)
        alpha *= active

        composite = (
            (1.0 - alpha[..., None]) * self._background
            + alpha[..., None] * coloured_foreground
        )

        # Preserve the terminal observation strongly; failure posture is often
        # the most useful part of a behavior trace.
        if self._last_frame is not None:
            terminal_difference = np.mean(
                np.abs(self._last_frame - self._background), axis=2
            )
            terminal_mask = np.clip(
                terminal_difference / max(self.motion_threshold * 2.0, 1.0),
                0.0,
                1.0,
            )
            terminal_tint = (
                (1.0 - self.tint_strength) * self._last_frame
                + self.tint_strength * np.array([240.0, 55.0, 45.0])
            )
            terminal_alpha = 0.55 * terminal_mask[..., None]
            composite = (1.0 - terminal_alpha) * composite + terminal_alpha * terminal_tint

        return np.clip(composite, 0.0, 255.0).astype(np.uint8)
