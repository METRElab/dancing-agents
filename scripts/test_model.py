import argparse
import yaml
import numpy as np
from sb3_contrib.ppo_recurrent import RecurrentPPO
from rl_env.synchronization_env import SynchronizationEnv


def run_test(model_path, config, tempo_override=None, jitter_override=None):
    env_config = config['environment']
    if tempo_override:
        env_config['min_tempo'] = env_config['max_tempo'] = tempo_override
    if jitter_override:
        env_config['jitter'] = jitter_override

    env = SynchronizationEnv(env_config)
    model = RecurrentPPO.load(model_path, env=env)
    obs, _ = env.reset()
    lstm_states = model.policy.initial_state(1)

    cues, taps = [], []
    done = False

    while not done:
        action, lstm_states = model.policy.predict(obs, state=lstm_states, episode_start=np.array([False]), deterministic=True)
        obs, _, done, _, _ = env.step(action)
        if env.current_step in env.cue_times:
            cues.append(env.current_step)
        if env.tap_buffer and env.tap_buffer[-1] not in taps:
            taps.append(env.tap_buffer[-1])

    cues_ms = np.array(cues) * env.dt * 1000
    taps_ms = np.array(taps) * env.dt * 1000
    asynchronies = []
    for tap in taps:
        nearest_cue = min(cues, key=lambda c: abs(c - tap))
        asynchrony = (tap - nearest_cue) * env.dt * 1000
        asynchronies.append(asynchrony)

    print("Cue times (ms):", cues_ms)
    print("Tap times (ms):", taps_ms)
    print("Tap-cue asynchronies (ms):", asynchronies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--tempo', type=float)
    parser.add_argument('--jitter', type=float)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    run_test(args.model, config, tempo_override=args.tempo, jitter_override=args.jitter)