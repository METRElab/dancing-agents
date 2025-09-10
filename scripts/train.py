import os
import yaml
import argparse
import gymnasium as gym
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.ppo_recurrent import RecurrentPPO
from utils.hidden_state_callback import HiddenStateRecorder
from rl_env.synchronization_env import SynchronizationEnv


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env(env_config):
    class WrappedSyncEnv(SynchronizationEnv):
        def __init__(self):
            super().__init__(env_config)
    return WrappedSyncEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    env_config = config['environment']
    rl_config = config['rl_algorithm']
    policy_kwargs = config.get('policy_kwargs', {})

    env_fn = make_env(env_config)
    env = gym.vector.SyncVectorEnv([lambda: env_fn()])

    model = RecurrentPPO(
        policy=rl_config.get('policy', 'MlpLstmPolicy'),
        env=env,
        verbose=1,
        learning_rate=rl_config.get('learning_rate', 3e-4),
        n_steps=env_config.get('n_steps', 128),
        batch_size=rl_config.get('batch_size', 32),
        gae_lambda=rl_config.get('gae_lambda', 0.95),
        gamma=rl_config.get('gamma', 0.99),
        n_epochs=rl_config.get('n_epochs', 10),
        ent_coef=rl_config.get('ent_coef', 0.01),
        policy_kwargs=policy_kwargs,
        tensorboard_log="logs/"
    )

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="models/",
                                             name_prefix="rl_model")
    
    hidden_callback = HiddenStateRecorder(
    save_path="logs/hidden_states.npy",
    save_freq=5000,
    verbose=1
)

    model.learn(
    total_timesteps=config.get('total_timesteps', 1e6),
    callback=[checkpoint_callback, hidden_callback]
)

    model.save("models/final_model")

if __name__ == '__main__':
    main()
