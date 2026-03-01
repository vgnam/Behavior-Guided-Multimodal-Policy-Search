"""
Periodic Frame Sampling for Vision-Guided Policy Search

Samples frames from an episode trajectory using a periodic (cycle) strategy:
 - first frame  (t = 0)
 - last frame   (t = T-1)
 - failure frame: last frame when episode ended early
 - periodic frames: t = P, 2P, 3P, ... (fixed interval)
"""

import numpy as np
from typing import List, Dict, Any


class FrameSampler:
    """
    Samples frames from a trajectory at regular periodic intervals.
    """

    def __init__(self, sample_period: int = 25):
        """
        Initialize the periodic frame sampler.

        Args:
            sample_period: P â€” capture a frame every P timesteps.
                           E.g. P=50 captures t=0, 50, 100, ... plus first/last.
        """
        self.sample_period = max(1, sample_period)

    def sample_frames(
        self,
        trajectory: List[Dict[str, Any]],
        terminated: bool,
        max_traj_length: int,
    ) -> Dict[str, List[int]]:
        """
        Return frame indices using the periodic strategy.

        Args:
            trajectory: List of dicts with keys 'state', 'action', 'reward', 'frame'
            terminated: Whether the episode ended early (failure)
            max_traj_length: Maximum trajectory length

        Returns:
            Dict with keys 'first', 'last', 'fail', 'periodic', 'all'.
        """
        if len(trajectory) == 0:
            return {'first': [], 'last': [], 'fail': [], 'periodic': [], 'all': []}

        T = len(trajectory)
        indices: Dict[str, List[int]] = {
            'first': [],
            'last': [],
            'fail': [],
            'periodic': [],
            'all': [],
        }

        # First frame
        indices['first'].append(0)

        # Last frame
        indices['last'].append(T - 1)

        # Failure frame (last frame of an early-terminated episode)
        if terminated and T - 1 < max_traj_length - 1:
            indices['fail'].append(T - 1)

        # Periodic frames (skip 0 and T-1 â€” already included)
        for t in range(self.sample_period, T - 1, self.sample_period):
            indices['periodic'].append(t)

        # Union, sorted
        all_set: set = set()
        for key in ('first', 'last', 'fail', 'periodic'):
            all_set.update(indices[key])
        indices['all'] = sorted(all_set)

        return indices

    def get_frames(
        self,
        trajectory: List[Dict[str, Any]],
        frame_indices: Dict[str, List[int]],
    ) -> List[Dict[str, Any]]:
        """
        Extract frame dicts at the sampled indices.

        Returns:
            List of dicts: {'timestep', 'state', 'action', 'reward', 'frame', 'frame_type'}
        """
        frames = []

        for idx in frame_indices['all']:
            if idx < len(trajectory):
                frame_data = trajectory[idx].copy()
                frame_data['timestep'] = idx

                types = []
                if idx in frame_indices['first']:    types.append('first')
                if idx in frame_indices['last']:     types.append('last')
                if idx in frame_indices['fail']:     types.append('fail')
                if idx in frame_indices['periodic']: types.append('periodic')

                frame_data['frame_type'] = ','.join(types)
                frames.append(frame_data)

        return frames

    def format_frames_summary(
        self,
        frames: List[Dict[str, Any]],
    ) -> str:
        """Generate a text summary of sampled frames for logging."""
        summary = f"Sampled Frames Summary (Total: {len(frames)})\n"
        summary += "=" * 60 + "\n"

        for frame in frames:
            summary += f"Timestep {frame['timestep']} [{frame['frame_type']}]:\n"
            summary += f"  State: {frame['state']}\n"
            summary += f"  Action: {frame['action']}\n"
            summary += f"  Reward: {frame['reward']:.4f}\n"
            summary += "-" * 60 + "\n"

        return summary
