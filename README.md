# Reinforcement Learning Framework for Sensorimotor Synchronization 

A lightweight framework for learning sensorimotor synchronization in a simulated finger-tapping experiment using reinforcement learning. 

## Features

- Recurrent PPO agent (LSTM) trained on metronome tapping task
- Configurable tempo, jitter, feedback, delay, and reward types
- Hidden state recording for PCA and neural analysis
- Simple training, testing, and analysis scripts

## Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train
```bash
python scripts/train.py --config config.yaml
```

### Analyze hidden states
```bash
python scripts/analyze_hidden.py --model models/final_model.zip --config config.yaml
```

### Test agent behavior
```bash
python scripts/test_model.py --model models/final_model.zip --config config.yaml --tempo 0.5 --jitter 0.01
```

## Citation

Yassaman Ommi, Matin Yousefabadi, Jonathan Cannon (2024). *Reinforcement-trained recurrent networks reproduce human beat-synchronization dynamics.*
