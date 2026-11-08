"""
Live training monitor: console status, CSV log, and a progress plot.

Tracks the things that actually diagnose this task:
  * mean episode reward (should climb toward 0)
  * max single-step reward (must stay <= 0; anything positive is a bug)
  * taps per episode (a silent agent taps 0; a healthy one ~10)
  * mean asynchrony in ms (the paper's agent is slightly negative)

Writes ``training_log.csv`` and ``training_progress.png`` into ``out_dir`` every
``plot_freq`` steps and once more at the end, so the plot is readable while the
run is still going. Uses the Agg backend, so it needs no display and no network.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TrainingMonitor(BaseCallback):
    def __init__(self, out_dir: str = "logs", plot_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.out_dir = out_dir
        self.plot_freq = plot_freq
        os.makedirs(out_dir, exist_ok=True)

        self.steps: list[int] = []
        self.ep_rewards: list[float] = []
        self.ep_taps: list[float] = []
        self.ep_async: list[float] = []

        self.max_step_reward = -np.inf
        self.n_positive_rewards = 0
        self.first_positive_at: int | None = None

    def _on_step(self) -> bool:
        # Every step: watch for the sign-flip bug reappearing
        for r in np.atleast_1d(self.locals["rewards"]):
            r = float(r)
            if r > self.max_step_reward:
                self.max_step_reward = r
            if r > 1e-9:
                self.n_positive_rewards += 1
                if self.first_positive_at is None:
                    self.first_positive_at = self.num_timesteps

        # On episode end: record the episode's summary
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue
            self.steps.append(self.num_timesteps)
            self.ep_rewards.append(float(ep["r"]))
            self.ep_taps.append(float(info.get("n_taps", 0)))
            self.ep_async.append(float(info.get("mean_asynchrony_ms", np.nan)))

        if self.plot_freq and self.n_calls % self.plot_freq == 0:
            self.save()
            if self.verbose:
                self._print_status()
        return True

    # output
    def _print_status(self) -> None:
        if not self.ep_rewards:
            print(f"[monitor] {self.num_timesteps:,} steps — no completed episodes yet")
            return
        k = min(50, len(self.ep_rewards))
        recent_async = [a for a in self.ep_async[-k:] if not np.isnan(a)]
        async_txt = f"{np.mean(recent_async):+6.1f} ms" if recent_async else "     n/a"
        flag = "  *** POSITIVE REWARD SEEN - BUG ***" if self.n_positive_rewards else ""
        print(
            f"[monitor] {self.num_timesteps:>9,} steps | "
            f"ep_reward {np.mean(self.ep_rewards[-k:]):+7.3f} | "
            f"taps/ep {np.mean(self.ep_taps[-k:]):5.2f} | "
            f"async {async_txt} | "
            f"max step r {self.max_step_reward:+.4f}{flag}"
        )

    def save(self) -> None:
        self._write_csv()
        self._write_plot()

    def _write_csv(self) -> None:
        path = os.path.join(self.out_dir, "training_log.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestep", "episode_reward", "taps", "mean_asynchrony_ms"])
            for row in zip(self.steps, self.ep_rewards, self.ep_taps, self.ep_async):
                w.writerow(row)

    @staticmethod
    def _rolling(y, frac=0.05, lo=10):
        n = max(lo, int(len(y) * frac))
        if len(y) < n or n < 2:
            return None
        return np.convolve(y, np.ones(n) / n, mode="valid"), n

    def _write_plot(self) -> None:
        if len(self.ep_rewards) < 2:
            return
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        panels = [
            (self.ep_rewards, "Episode reward", "0 = perfect sync", 0.0),
            (self.ep_taps, "Taps per episode", "paper: ~10", None),
            (self.ep_async, "Mean asynchrony (ms)", "paper: slightly negative", 0.0),
        ]
        for ax, (y, title, note, ref) in zip(axes, panels):
            y = np.asarray(y, dtype=float)
            ax.plot(self.steps, y, lw=0.6, alpha=0.3, color="#4C72B0")
            roll = self._rolling(y)
            if roll is not None:
                sm, n = roll
                ax.plot(self.steps[n - 1:], sm, lw=1.8, color="#1F3C73",
                        label=f"rolling mean ({n} eps)")
                ax.legend(loc="lower right", fontsize=8, frameon=False)
            if ref is not None:
                ax.axhline(ref, color="#B4472F", lw=0.9, ls="--", alpha=0.8)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(note, fontsize=8, loc="right", color="#666")
            ax.grid(alpha=0.15)
        axes[-1].set_xlabel("training timesteps")
        status = (f"max step reward {self.max_step_reward:+.4f}"
                  + (f"  —  {self.n_positive_rewards} POSITIVE rewards, first at "
                     f"{self.first_positive_at:,} steps" if self.n_positive_rewards
                     else "  —  no positive rewards (correct)"))
        fig.suptitle(f"next+interval — {self.num_timesteps:,} steps\n{status}",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(self.out_dir, "training_progress.png"), dpi=130)
        plt.close(fig)

    def _on_training_end(self) -> None:
        self.save()
        if self.verbose:
            print("\n" + "=" * 62)
            self._print_status()
            print(f"  {len(self.ep_rewards):,} episodes completed")
            print(f"  plot -> {self.out_dir}/training_progress.png")
            print(f"  csv  -> {self.out_dir}/training_log.csv")
            print("=" * 62)
