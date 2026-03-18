# PLaT: Latent Space Reasoning Framework

This is the repo for ["Latent Chain-of-Thought as Planning: Decoupling Reasoning from Verbalization"](https://arxiv.org/abs/2601.21358).

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
plat/
├── config/          # Configuration files (YAML)
├── models/         # Model definitions (PLaTModel, CoTModel)
├── data/           # Data processing
├── engine/         # Training engine
├── scripts/        # Training and testing scripts
└── utils/         # Utility functions
```

## Quick Start

### Training

```bash
# CoT SFT Training
python scripts/train_sft.py --config config/cot-gpt2.yaml

# PLaT Training
torchrun --nproc_per_node=2 scripts/train_plat.py --config config/plat-gpt2.yaml

# RL Training
torchrun --nproc_per_node=2 scripts/train_rl.py --config config/plat-rl-gpt2.yaml --model_path path/to/plat_checkpoint
```

### Testing

```bash
python scripts/test_plat.py --model_path path/to/checkpoint
```

## Models

- **PLaTModel**: Latent space reasoning model with planner and decoder
- **CoTModel**: Chain-of-Thought model for SFT baseline

## Configuration

See `config/` for example YAML configuration files:
- `cot-gpt2.yaml`: CoT SFT training config
- `plat-gpt2.yaml`: PLaT training config
- `plat-rl-gpt2.yaml`: RL training config

