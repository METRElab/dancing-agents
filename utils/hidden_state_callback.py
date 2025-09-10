import os
import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class HiddenStateRecorder(BaseCallback):
    def __init__(self, save_path: str, save_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.hidden_states = []
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def _on_step(self) -> bool:
        # Only works if model has lstm states
        if hasattr(self.model.policy, 'lstm_hidden_states'):
            lstm_state = self.model.policy.lstm_hidden_states[0].squeeze().detach().cpu().numpy()
            self.hidden_states.append(lstm_state)

        if self.n_calls % self.save_freq == 0:
            np.save(self.save_path, np.array(self.hidden_states))
            if self.verbose:
                print(f"[HiddenStateRecorder] Saved {len(self.hidden_states)} states to {self.save_path}")

        return True

    def _on_training_end(self) -> None:
        np.save(self.save_path, np.array(self.hidden_states))
        if self.verbose:
            print(f"[HiddenStateRecorder] Final save: {len(self.hidden_states)} states")
