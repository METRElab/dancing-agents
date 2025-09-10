import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sb3_contrib.ppo_recurrent import RecurrentPPO
from rl_env.synchronization_env import SynchronizationEnv


def collect_hidden_states(model_path, config, n_trials=100):
    env = SynchronizationEnv(config['environment'])
    model = RecurrentPPO.load(model_path, env=env)

    lstm_states = model.policy.initial_state(1)
    obs, _ = env.reset()

    all_hidden = []
    for _ in range(n_trials):
        with torch.no_grad():
            action, lstm_states = model.policy.predict(
                obs,
                state=lstm_states,
                episode_start=torch.tensor([False]),
                deterministic=True,
                return_latent=True
            )
        latent = lstm_states[0].squeeze().cpu().numpy()
        all_hidden.append(latent)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
            lstm_states = model.policy.initial_state(1)

    return np.array(all_hidden)


def apply_pca_and_save(hidden_states, csv_path):
    pca = PCA(n_components=3)
    coords = pca.fit_transform(hidden_states)
    df = pd.DataFrame(coords, columns=['PC1', 'PC2', 'PC3'])
    df.to_csv(csv_path, index=False)
    print(f"Saved PCA coords to {csv_path}")


if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--out', type=str, default="pca_coords.csv")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    hidden_states = collect_hidden_states(args.model, config)
    apply_pca_and_save(hidden_states, args.out)
