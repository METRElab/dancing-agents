import gymnasium as gym
from gymnasium import spaces
import numpy as np


def _reward_tokens(reward_type: str):
    return [t.strip().lower() for t in reward_type.replace(" ", "").split("+") if t.strip()]


class SynchronizationEnv(gym.Env):
    """Metronome-tapping sensorimotor synchronization task (Ommi, Yousefabadi & Cannon).

    On each episode the agent hears a rhythmic sequence of binary metronome cues and
    learns to tap in synchrony. The first ``entrainment_beats`` cues are non-evaluative,
    letting the agent entrain to the rhythm before synchronization is assessed (the paper
    begins assessment after the third cue).

    Observation: a single binary channel that is 1 on the step of a cue onset and 0
    otherwise (Methods: "binary metronome signal to each LSTM unit").

    Time base: ``dt`` is the environment step in seconds (0.01 = 10 ms; the paper's
    timestep resolution).

    Inter-onset interval (IOI): drawn uniformly from ``[min_ioi, max_ioi]`` seconds at
    the start of each episode (paper range: 450-750 ms), setting the tempo for that trial.

    Motor execution delay: ``tap_delay`` (seconds) is the fixed lag between initiating a
    tap (action 1) and the tap landing for reward (paper: 50 ms), requiring predictive
    timing.

    Reward (all rewards are non-positive; 0 = perfect synchronization; time in SECONDS).
    Rewards are delivered at the moment of tap landing, not action selection. The reward
    type is a "+"-joined token string selecting one of the four schemes in the paper:
      * ``nearest``       -> symmetric:  r_sym  = -|t_tap - t_nearest|            (Eq. 1)
      * ``next``          -> asymmetric: r_asym =  (t_tap - t_next)               (Eq. 2)
      * ``nearest+interval`` -> symmetric + interval                              (Eq. 3)
      * ``next+interval``    -> asymmetric + interval (most successful agent)     (Eq. 5)
    with the interval matching term
        r_int = -0.2 * min((ITI - IOI)^2, 0.16 * IOI)                            (Eq. 4)

    Missed-cue penalty: a cue that passes with no tap within IOI/2 incurs, at the close of
    that window, a tempo-scaling penalty
        r_missed = -IOI*(IOI - dt)/2 + dt                                        (Eq. 6)
    Entrainment cues are exempt from this penalty.

    No-tap penalty: an episode with no taps at all incurs r_notap = -1.

    Jitter: ``jitter`` is the standard deviation of cue-time displacement in SECONDS
    (e.g. 0.03 = 30 ms), applied as mean-zero Gaussian noise to the audible cues
    (paper test jitter: 30/40/50 ms). It is absolute (tempo-independent).

    Perturbation: setting ``perturb_beat`` (1-indexed cue number) and ``perturb`` (seconds)
    shifts a single audible cue, used to elicit error-correction responses (paper: the 6th
    cue shifted by 20 ms).
    """

    def __init__(self, config):
        super().__init__()

        self.dt = float(config.get("dt", 0.01))

        # IOI range in seconds (paper: 450-750 ms). Accept legacy min_tempo/max_tempo keys.
        self.min_ioi = float(config.get("min_ioi", config.get("min_tempo", 0.45)))
        self.max_ioi = float(config.get("max_ioi", config.get("max_tempo", 0.75)))

        tap_delay_s = float(config.get("tap_delay", 0.05))
        self.tap_delay = max(0, int(round(tap_delay_s / self.dt)))

        # Cue-time jitter SD in SECONDS (absolute, tempo-independent).
        self.jitter = float(config.get("jitter", 0.0))

        self.experiment_type = config.get("experiment_type", "synchronization")
        self.reward_type = config.get("reward_type", "next+interval")
        self.sync_beats = int(config.get("sync_beats", 10))
        self.continuation_beats = int(config.get("continuation_beats", 0))

        # Non-evaluative entrainment phase (paper: first three cues are non-evaluative).
        self.entrainment_beats = int(config.get("entrainment_beats", 3))

        # Interval-matching term (Eq. 4): weight 0.2, squared error capped at 0.16*IOI.
        self.interval_weight = float(config.get("interval_weight", 0.2))

        # Missed-cue penalty: a cue is "missed" if no tap lands within
        # missed_tap_window_frac * IOI (radius). Default radius = IOI/2 (paper).
        self.use_missed_penalty = bool(config.get("use_missed_penalty", True))
        self.missed_tap_window_frac = float(config.get("missed_tap_window_frac", 0.5))

        # Whole-episode no-tap penalty (paper: r_notap = -1).
        self.notap_penalty = float(config.get("notap_penalty", -1.0))

        # Single-cue perturbation (paper: 6th cue shifted by 20 ms).
        self.perturb_beat = int(config.get("perturb_beat", 0))  # 1-indexed; 0 = off
        self.perturb = float(config.get("perturb", 0.0))         # shift in seconds

        # Observation is the binary metronome signal (paper).
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.ioi = None
        self.cue_times = []
        self.virtual_cue_times = []
        self.tap_buffer = []
        self.current_step = 0
        self.timestep_limit = None
        self.pending_tap = None

        self._missed_window = 0
        self._eval_cutoff = 0
        self._cue_hit = []
        self._cue_penalized = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        ioi_s = float(self.np_random.uniform(self.min_ioi, self.max_ioi))
        self.ioi = max(1, int(round(ioi_s / self.dt)))
        total_beats = self.sync_beats + self.continuation_beats
        self.timestep_limit = self.ioi * (total_beats + 2)

        # Audible sync cues at nominal grid positions.
        self.cue_times = [i * self.ioi for i in range(self.sync_beats)]

        # Optional single-cue perturbation (e.g. 6th cue shifted by 20 ms).
        if self.perturb_beat > 0 and self.perturb != 0.0:
            idx = self.perturb_beat - 1
            if 0 <= idx < len(self.cue_times):
                self.cue_times[idx] = max(
                    0, int(round(self.cue_times[idx] + self.perturb / self.dt))
                )

        # Mean-zero Gaussian cue-time jitter (absolute SD in seconds).
        if self.jitter > 0:
            jitter_std = self.jitter / self.dt
            noise = self.np_random.normal(0, jitter_std, size=len(self.cue_times))
            self.cue_times = np.clip(
                (np.array(self.cue_times, dtype=np.float64) + noise).astype(int), 0, None
            ).tolist()

        # Continuation beats are silent (not in cue_times) but still scored against the grid.
        self.virtual_cue_times = list(self.cue_times) + [
            i * self.ioi for i in range(self.sync_beats, total_beats)
        ]

        self.tap_buffer = []
        self.current_step = 0
        self.pending_tap = None

        self._missed_window = max(1, int(round(self.missed_tap_window_frac * self.ioi)))
        # Taps before this step belong to the entrainment phase and are not evaluated.
        self._eval_cutoff = max(0, self.entrainment_beats * self.ioi - self._missed_window)

        n_cues = len(self.virtual_cue_times)
        self._cue_hit = [False] * n_cues
        self._cue_penalized = [False] * n_cues

        return self._get_obs(), {}

    def _async_reward(self, tap_time, tokens):
        if not self.virtual_cue_times:
            return -1.0

        nearest = min(self.virtual_cue_times, key=lambda c: abs(c - tap_time))
        next_cues = [c for c in self.virtual_cue_times if c >= tap_time]
        next_cue = next_cues[0] if next_cues else nearest

        if "nearest" in tokens:
            r_async = -abs(tap_time - nearest) * self.dt
        elif "next" in tokens:
            r_async = -(next_cue - tap_time) * self.dt
        else:
            r_async = 0.0
        return float(r_async)

    def _interval_reward(self, tap_time, prev_tap, tokens):
        if prev_tap is None or "interval" not in tokens:
            return 0.0

        ITI = tap_time - prev_tap
        err_s = (ITI - self.ioi) * self.dt
        ioi_s = self.ioi * self.dt
        cap = 0.16 * ioi_s  # Eq. 4
        r_interval = -self.interval_weight * min(err_s ** 2, cap)
        return float(r_interval)

    def _register_tap_for_missed(self, tap_time):
        for i, c in enumerate(self.virtual_cue_times):
            if abs(tap_time - c) <= self._missed_window:
                self._cue_hit[i] = True

    def _missed_cue_penalties(self):
        extra = 0.0
        if not self.use_missed_penalty:
            return extra
        # Tempo-scaling missed penalty (Eq. 6), IOI in seconds.
        ioi_s = self.ioi * self.dt
        penalty = -(ioi_s * (ioi_s - self.dt)) / 2.0 + self.dt
        t = self.current_step
        for i, c in enumerate(self.virtual_cue_times):
            if self._cue_penalized[i]:
                continue
            if i < self.entrainment_beats:  # non-evaluative entrainment cue
                self._cue_penalized[i] = True
                continue
            if t < c + self._missed_window:  # window still open
                continue
            if not self._cue_hit[i]:
                extra += penalty
            self._cue_penalized[i] = True
        return extra

    def _episode_timing_metrics(self):
        taps = [t for t in self.tap_buffer if t >= self._eval_cutoff]
        if not taps or not self.virtual_cue_times:
            return {}
        async_ms = []
        for tap in taps:
            nearest_cue = min(self.virtual_cue_times, key=lambda c: abs(c - tap))
            async_ms.append((tap - nearest_cue) * self.dt * 1000.0)
        arr = np.asarray(async_ms, dtype=np.float64)
        return {
            "mean_abs_asynchrony_ms": float(np.mean(np.abs(arr))),
            "mean_asynchrony_ms": float(np.mean(arr)),
            "std_asynchrony_ms": float(np.std(arr)),
            "n_taps": int(len(taps)),
        }

    def step(self, action):
        reward = 0.0
        done = False
        info = {}

        if action == 1:
            self.pending_tap = self.current_step + self.tap_delay

        if self.pending_tap is not None and self.pending_tap == self.current_step:
            self.tap_buffer.append(self.current_step)
            self._register_tap_for_missed(self.current_step)
            reward += self._compute_reward(self.current_step)
            self.pending_tap = None

        # Missed-cue penalties for any evaluated cue whose window has just closed.
        reward += self._missed_cue_penalties()

        self.current_step += 1

        if self.current_step >= self.timestep_limit:
            done = True
            # Whole-episode no-tap penalty: a silent episode is the worst outcome.
            if len(self.tap_buffer) == 0:
                reward += self.notap_penalty
            info.update(self._episode_timing_metrics())

        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        pulse = 1.0 if self.current_step in self.cue_times else 0.0
        return np.array([pulse], dtype=np.float32)

    def _compute_reward(self, tap_time):
        if not self.virtual_cue_times:
            return -1.0
        # Non-evaluative entrainment phase: taps before the cutoff are not scored.
        if tap_time < self._eval_cutoff:
            return 0.0

        tokens = _reward_tokens(self.reward_type)
        prev_tap = self.tap_buffer[-2] if len(self.tap_buffer) > 1 else None

        r_async = self._async_reward(tap_time, tokens)
        r_interval = self._interval_reward(tap_time, prev_tap, tokens)
        return r_async + r_interval

    def render(self):
        pass

    def close(self):
        pass
