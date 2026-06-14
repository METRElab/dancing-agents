# Reinforcement-Trained Recurrent Networks for Beat Synchronization

Code accompanying *Reinforcement-trained recurrent networks reproduce human
beat-synchronization dynamics* (Ommi, Yousefabadi & Cannon). The repository trains an
LSTM-based recurrent agent to tap in time with a metronome under reinforcement learning,
and provides tools to test its synchronization behavior and inspect its internal neural
dynamics.

## Method overview

**Task.** A metronome-tapping synchronization task. On each episode the agent hears a
rhythmic sequence of binary metronome cues and learns to tap in synchrony. The first
three cues are **non-evaluative**: they let the agent entrain to the rhythm before
synchronization is assessed (assessment begins after the third cue).

- Timestep resolution: `dt = 0.01 s` (10 ms).
- Inter-onset interval (IOI): drawn uniformly from **450–750 ms** at the start of each
  episode, setting the tempo for that trial.
- Metronome signal: binary — 1 on a cue onset, 0 otherwise.
- Actions: tap (1) or no tap (0).
- Motor execution delay: a fixed **50 ms** lag between tap initiation and tap landing,
  requiring predictive timing.

**Agent.** A recurrent neural network with a single LSTM layer of **128 units**. The
binary metronome signal is fed to the LSTM; its output drives a perceptron unit producing
a tap probability `P_tap`, and the agent taps when `P_tap > 0.5`. Training uses recurrent
Proximal Policy Optimization (PPO) from `sb3-contrib`.

**Reward schemes.** All rewards are non-positive (0 = perfect synchronization) and are
delivered at the moment of tap landing. Four schemes combine three ingredients —
symmetric vs. asymmetric tap/cue penalties and an optional interval-matching term:

| `reward_type` | Name | Asynchrony term |
|:--|:--|:--|
| `nearest` | Symmetric | `r_sym  = -|t_tap - t_nearest|` (Eq. 1) |
| `next` | Asymmetric | `r_asym =  (t_tap - t_next)` (Eq. 2) |
| `nearest+interval` | Symmetric + Interval | `r_sym + r_int` (Eq. 3) |
| `next+interval` | Asymmetric + Interval | `r_asym + r_int` (Eq. 5) |

The interval-matching term rewards inter-tap intervals (ITI) close to the IOI:

```
r_int = -0.2 * min((ITI - IOI)^2, 0.16 * IOI)        (Eq. 4)
```

Two additional penalties discourage inaction:

- **Missed cue** (Eq. 6): a cue that passes with no tap within `IOI/2` incurs, at the
  close of that window, a tempo-scaling penalty `r_missed = -IOI*(IOI - dt)/2 + dt`.
  Entrainment cues are exempt.
- **No taps**: an episode with no taps at all incurs `r_notap = -1`.

**Evaluation.** After training, agents are tested under steady, perturbed, and jittered
metronomes:

- *Perturbed*: a single cue (the 6th) is shifted (paper: 20 ms) to elicit error
  correction.
- *Jittered*: every cue is displaced by mean-zero Gaussian jitter (paper: 30, 40, or
  50 ms SD).
- *Neural*: PCA is applied to LSTM hidden-unit trajectories across 1000 test trials,
  retaining the top three components.

**Synchronization–continuation task.** Setting `continuation_beats > 0` plays the
metronome during a synchronization phase (e.g. 10 cues), then goes silent for a
continuation phase (e.g. 6 silent beats), during which the agent continues tapping at the
internalized tempo. The same trained `next+interval` weights are used with no retraining.

## Configuration

Key environment parameters in `configs/config.yaml`:

| Parameter | Description |
|:--|:--|
| `min_ioi`, `max_ioi` | IOI range in seconds (default 0.45–0.75) |
| `tap_delay` | Motor execution delay in seconds (default 0.05) |
| `reward_type` | `nearest` \| `next` \| `nearest+interval` \| `next+interval` |
| `sync_beats` | Number of audible metronome cues |
| `entrainment_beats` | Leading non-evaluative cues (default 3) |
| `continuation_beats` | Silent beats after cues stop (0 = sync-only) |
| `jitter` | Cue-time Gaussian jitter SD in seconds (e.g. 0.03 = 30 ms) |
| `perturb_beat`, `perturb` | 1-indexed cue to shift, and shift in seconds |

## Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train
```bash
python scripts/train.py --config configs/config.yaml
```

### Test agent behavior

By default the test script aggregates over **100 evaluation episodes** at randomly
sampled tempi (matching the paper) and reports mean ± SD asynchrony. Use `--episodes 1`
for a single-trial printout of cue/tap times.

```bash
# 100 episodes, steady metronome, mean +/- SD asynchrony
python scripts/test_model.py --model models/final_model.zip --config configs/config.yaml

# Single-trial behavior printout at a fixed 600 ms IOI
python scripts/test_model.py --model models/final_model.zip --config configs/config.yaml --ioi 0.6 --episodes 1

# Perturbed metronome: shift the 6th cue by 20 ms
python scripts/test_model.py --model models/final_model.zip --config configs/config.yaml --perturb-ms 20

# Jittered metronome: 30 ms SD
python scripts/test_model.py --model models/final_model.zip --config configs/config.yaml --jitter 0.03
```
When `continuation_beats > 0` (and `--episodes 1`), the test script also reports
continuation-phase statistics (tap count, inter-tap intervals, comparison to expected IOI).

### Analyze hidden states
```bash
python scripts/analyze_hidden.py --model models/final_model.zip --config configs/config.yaml
```

## Citation

Yassaman Ommi, Matin Yousefabadi, Jonathan Cannon. *Reinforcement-trained recurrent
networks reproduce human beat-synchronization dynamics.*
