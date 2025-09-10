import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SynchronizationEnv(gym.Env):
    """
    Custom Gymnasium environment for sensorimotor synchronization.
    The agent must learn to tap in synchrony with metronomic cues under various conditions.
    """

    def __init__(self, config):
        super().__init__()

        # Configuration
        self.dt = config.get("dt", 0.01) # in seconds 
        self.min_tempo = config.get("min_tempo", 0.4)
        self.max_tempo = config.get("max_tempo", 0.7)
        self.tap_delay = int(config.get("tap_delay", 0.05) / self.dt)
        self.jitter = config.get("jitter", 0.0)
        self.experiment_type = config.get("experiment_type", "synchronization")
        self.reward_type = config.get("reward_type", "next+interval")

        # Gym spaces
        self.action_space = spaces.Discrete(2)  # 0: no tap, 1: tap
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Environment state
        self.ioi = None
        self.cue_times = []
        self.tap_buffer = []
        self.current_step = 0
        self.timestep_limit = None
        self.pending_tap = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        tempo = self.np_random.uniform(self.min_tempo, self.max_tempo)
        self.ioi = int(tempo / self.dt)
        self.timestep_limit = self.ioi * 12  # 15 beat trials

        self.cue_times = [i * self.ioi for i in range(0, 10)]
        if self.jitter > 0:
            jitter = self.np_random.normal(0, self.jitter / self.dt, size=len(self.cue_times))
            self.cue_times = (np.array(self.cue_times) + jitter).astype(int)

        self.tap_buffer = []
        self.current_step = 0
        self.pending_tap = None

        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        done = False
        info = {}

        # Handle tap action
        if action == 1:
            self.pending_tap = self.current_step + self.tap_delay

        # Execute delayed tap
        if self.pending_tap is not None and self.pending_tap == self.current_step:
            self.tap_buffer.append(self.current_step)
            reward = self._compute_reward(self.current_step)
            self.pending_tap = None

        self.current_step += 1

        if self.current_step >= self.timestep_limit:
            done = True

        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        return np.array([1.0 if self.current_step in self.cue_times else 0.0], dtype=np.float32)

    def _compute_reward(self, tap_time):
        if not self.cue_times:
            return -1.0

        nearest = min(self.cue_times, key=lambda c: abs(c - tap_time))
        next_cues = [c for c in self.cue_times if c >= tap_time]
        next_cue = next_cues[0] if next_cues else nearest
        prev_tap = self.tap_buffer[-2] if len(self.tap_buffer) > 1 else None

        # Asynchrony rewards
        if "nearest" in self.reward_type:
            r_async = -abs(tap_time - nearest)
        elif "next" in self.reward_type:
            r_async = -(next_cue - tap_time)
        else:
            r_async = 0.0

        # Interval matching reward
        if "interval" in self.reward_type and prev_tap is not None:
            ITI = tap_time - prev_tap
            r_interval = -0.2 * min((ITI - self.ioi) ** 2, 0.16 * self.ioi)
        else:
            r_interval = 0.0

        return r_async + r_interval

    def render(self):
        pass

    def close(self):
        pass
