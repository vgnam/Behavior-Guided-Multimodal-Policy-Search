"""
Critical Frame Sampling for Vision-Guided Policy Search

This module implements the critical frame sampling strategy defined in Eq. (2):
I_crit = I_init ∪ I_term ∪ I_fail ∪ I_trans

Where:
- I_init: Initial state frame  
- I_term: Terminal state frame
- I_fail: Failure states (early termination)
- I_trans: Transition states with significant reward changes
"""

import numpy as np
from typing import List, Dict, Tuple, Any


class CriticalFrameSampler:
    """
    Samples critical frames from a trajectory based on pivotal moments.
    """
    
    def __init__(
        self, 
        reward_change_threshold: float = 0.1,
        sample_strategy: str = 'reward',
        num_middle_frames: int = 2
    ):
        """
        Initialize the critical frame sampler.
        
        Args:
            reward_change_threshold: Threshold δ for detecting transition states
                                    where |ΔR_t| > δ
            sample_strategy: Strategy for sampling transition states
                           - 'reward': Based on reward changes (default)
                           - 'uniform': Uniformly sample middle frames
                           - 'state': Based on state changes (for constant-reward envs)
            num_middle_frames: Number of middle frames to sample for uniform/state strategy
        """
        self.delta = reward_change_threshold
        self.sample_strategy = sample_strategy
        self.num_middle_frames = num_middle_frames
        
    def sample_critical_frames(
        self, 
        trajectory: List[Dict[str, Any]],
        terminated: bool,
        max_traj_length: int
    ) -> Dict[str, List[int]]:
        """
        Extract critical frame indices from a trajectory.
        
        Args:
            trajectory: List of trajectory steps, each containing:
                       {'state': np.array, 'action': np.array, 'reward': float, 'frame': np.array}
            terminated: Whether episode terminated early (failure)
            max_traj_length: Maximum trajectory length
            
        Returns:
            Dictionary with keys: 'init', 'term', 'fail', 'trans', 'all'
            Each containing list of frame indices
        """
        if len(trajectory) == 0:
            return {'init': [], 'term': [], 'fail': [], 'trans': [], 'all': []}
        
        critical_indices = {
            'init': [],
            'term': [],
            'fail': [],
            'trans': [],
            'all': []
        }
        
        # I_init: Initial state (t=0)
        critical_indices['init'].append(0)
        
        # I_term: Terminal state (last frame)
        terminal_idx = len(trajectory) - 1
        critical_indices['term'].append(terminal_idx)
        
        # I_fail: Failure states (early termination)
        if terminated and terminal_idx < max_traj_length - 1:
            critical_indices['fail'].append(terminal_idx)
        
        # I_trans: Transition states - strategy depends on configuration
        if self.sample_strategy == 'reward':
            # Original: Based on reward changes |ΔR_t| > δ
            for t in range(1, len(trajectory)):
                reward_t = trajectory[t]['reward']
                reward_t_prev = trajectory[t-1]['reward']
                delta_reward = abs(reward_t - reward_t_prev)
                
                if delta_reward > self.delta:
                    critical_indices['trans'].append(t)
                    
        elif self.sample_strategy == 'uniform':
            # For constant-reward environments (e.g., CartPole)
            # Sample uniformly spaced middle frames
            if len(trajectory) > 2:
                middle_indices = np.linspace(
                    1, 
                    len(trajectory) - 2, 
                    min(self.num_middle_frames, len(trajectory) - 2)
                ).astype(int)
                critical_indices['trans'].extend(middle_indices.tolist())
                
        elif self.sample_strategy == 'state':
            # Sample based on state changes (for environments with minimal reward variance)
            # Detect significant state changes
            for t in range(1, len(trajectory)):
                state_t = trajectory[t]['state']
                state_t_prev = trajectory[t-1]['state']
                
                # Compute normalized state change
                state_change = np.linalg.norm(state_t - state_t_prev)
                
                # Use adaptive threshold: top percentile of state changes
                if t > 1:  # Need at least 2 samples
                    all_changes = []
                    for i in range(1, t + 1):
                        s_change = np.linalg.norm(
                            trajectory[i]['state'] - trajectory[i-1]['state']
                        )
                        all_changes.append(s_change)
                    
                    threshold = np.percentile(all_changes, 75)  # Top 25%
                    if state_change > threshold:
                        critical_indices['trans'].append(t)
        
        # Combine all critical indices (union) and sort
        all_critical = set()
        for key in ['init', 'term', 'fail', 'trans']:
            all_critical.update(critical_indices[key])
        
        critical_indices['all'] = sorted(list(all_critical))
        
        return critical_indices
    
    def get_critical_frames(
        self,
        trajectory: List[Dict[str, Any]],
        critical_indices: Dict[str, List[int]]
    ) -> List[Dict[str, Any]]:
        """
        Extract the actual frames at critical indices.
        
        Args:
            trajectory: Full trajectory
            critical_indices: Dictionary of critical frame indices
            
        Returns:
            List of frames at critical timesteps, each containing:
            {'timestep': int, 'state': np.array, 'action': np.array, 
             'reward': float, 'frame': np.array, 'frame_type': str}
        """
        critical_frames = []
        
        for idx in critical_indices['all']:
            if idx < len(trajectory):
                frame_data = trajectory[idx].copy()
                frame_data['timestep'] = idx
                
                # Determine frame type(s)
                frame_types = []
                if idx in critical_indices['init']:
                    frame_types.append('init')
                if idx in critical_indices['term']:
                    frame_types.append('term')
                if idx in critical_indices['fail']:
                    frame_types.append('fail')
                if idx in critical_indices['trans']:
                    frame_types.append('trans')
                
                frame_data['frame_type'] = ','.join(frame_types)
                critical_frames.append(frame_data)
        
        return critical_frames
    
    def format_critical_frames_summary(
        self,
        critical_frames: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a text summary of critical frames for logging.
        
        Args:
            critical_frames: List of critical frame dictionaries
            
        Returns:
            Formatted string summary
        """
        summary = f"Critical Frames Summary (Total: {len(critical_frames)})\n"
        summary += "=" * 60 + "\n"
        
        for frame in critical_frames:
            summary += f"Timestep {frame['timestep']} [{frame['frame_type']}]:\n"
            summary += f"  State: {frame['state']}\n"
            summary += f"  Action: {frame['action']}\n"
            summary += f"  Reward: {frame['reward']:.4f}\n"
            summary += "-" * 60 + "\n"
        
        return summary
