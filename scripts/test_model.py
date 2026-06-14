import argparse
import yaml
import numpy as np
from sb3_contrib.ppo_recurrent import RecurrentPPO
from rl_env.synchronization_env import SynchronizationEnv


def build_env_config(config, ioi_override=None, jitter_override=None,
                     perturb_ms=None, perturb_beat=6):
    env_config = dict(config['environment'])
    if ioi_override:
        env_config['min_ioi'] = env_config['max_ioi'] = ioi_override
    if jitter_override is not None:
        env_config['jitter'] = jitter_override
    if perturb_ms is not None:
        # Perturbed metronome: shift a single cue (paper shifts the 6th cue by 20 ms).
        env_config['perturb_beat'] = perturb_beat
        env_config['perturb'] = perturb_ms / 1000.0
    return env_config


def run_episode(env, model):
    """Run one evaluation episode and return its taps, cues, and asynchronies (ms)."""
    obs, _ = env.reset()
    lstm_states = model.policy.initial_state(1)

    cues, taps = [], []
    done = False
    while not done:
        action, lstm_states = model.policy.predict(
            obs, state=lstm_states, episode_start=np.array([False]), deterministic=True
        )
        obs, _, done, _, _ = env.step(action)
        if env.current_step in env.cue_times:
            cues.append(env.current_step)
        if env.tap_buffer and env.tap_buffer[-1] not in taps:
            taps.append(env.tap_buffer[-1])

    virtual_cues = env.virtual_cue_times
    asynchronies = []
    for tap in taps:
        nearest_cue = min(virtual_cues, key=lambda c: abs(c - tap))
        asynchronies.append((tap - nearest_cue) * env.dt * 1000)

    # Evaluated taps only (after the non-evaluative entrainment phase).
    eval_async = [a for a, t in zip(asynchronies, taps) if t >= env._eval_cutoff]
    return {
        "cues": cues, "taps": taps,
        "asynchronies": asynchronies, "eval_async": eval_async,
    }


def print_single_trial(env, result):
    print("Cue times (ms):", np.array(result["cues"]) * env.dt * 1000)
    print("Tap times (ms):", np.array(result["taps"]) * env.dt * 1000)
    print("Tap-cue asynchronies (ms):", result["asynchronies"])
    if result["eval_async"]:
        print(f"Mean asynchrony (evaluated, ms): {np.mean(result['eval_async']):.1f}")

    if env.continuation_beats > 0:
        taps = result["taps"]
        last_cue = max(env.cue_times)
        sync_taps = [t for t in taps if t <= last_cue + env.ioi // 2]
        cont_taps = [t for t in taps if t > last_cue + env.ioi // 2]
        cont_itis = np.diff(cont_taps) * env.dt * 1000 if len(cont_taps) > 1 else []
        print("\n--- Continuation phase ---")
        print(f"Sync taps: {len(sync_taps)}, Continuation taps: {len(cont_taps)}")
        print(f"Continuation ITIs (ms): {list(cont_itis)}")
        print(f"Expected IOI (ms): {env.ioi * env.dt * 1000:.1f}")


def run_test(model_path, config, ioi_override=None, jitter_override=None,
             perturb_ms=None, perturb_beat=6, episodes=100):
    env_config = build_env_config(config, ioi_override, jitter_override,
                                  perturb_ms, perturb_beat)
    env = SynchronizationEnv(env_config)
    model = RecurrentPPO.load(model_path, env=env)

    if episodes <= 1:
        print_single_trial(env, run_episode(env, model))
        return

    # Aggregate over N evaluation episodes at randomly sampled tempi (paper: 100 per seed).
    per_episode_mean = []
    all_eval_async = []
    for _ in range(episodes):
        result = run_episode(env, model)
        if result["eval_async"]:
            per_episode_mean.append(np.mean(result["eval_async"]))
            all_eval_async.extend(result["eval_async"])

    print(f"Evaluation over {episodes} episodes "
          f"(IOI {env.min_ioi*1000:.0f}-{env.max_ioi*1000:.0f} ms"
          f"{'' if jitter_override is None else f', jitter {jitter_override*1000:.0f} ms'}"
          f"{'' if perturb_ms is None else f', cue {perturb_beat} +{perturb_ms:.0f} ms'}):")
    if not per_episode_mean:
        print("  No evaluated taps recorded.")
        return
    arr = np.asarray(all_eval_async)
    print(f"  Mean asynchrony:        {np.mean(per_episode_mean):+.1f} ms "
          f"(SD across episodes {np.std(per_episode_mean):.1f})")
    print(f"  Mean abs asynchrony:    {np.mean(np.abs(arr)):.1f} ms")
    print(f"  Pooled tap SD:          {np.std(arr):.1f} ms  (n={len(arr)} taps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ioi', type=float, help='Fixed IOI in seconds (e.g. 0.6)')
    parser.add_argument('--jitter', type=float, help='Cue jitter SD in seconds (e.g. 0.03)')
    parser.add_argument('--perturb-ms', type=float, dest='perturb_ms',
                        help='Shift a single cue by this many ms (paper: 20)')
    parser.add_argument('--perturb-beat', type=int, default=6, dest='perturb_beat',
                        help='1-indexed cue to perturb (paper: 6th)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes (paper: 100; use 1 for a '
                             'single-trial printout)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    run_test(args.model, config, ioi_override=args.ioi, jitter_override=args.jitter,
             perturb_ms=args.perturb_ms, perturb_beat=args.perturb_beat,
             episodes=args.episodes)
