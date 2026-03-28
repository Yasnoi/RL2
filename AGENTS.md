# AGENTS.md - AI Assistant Guide

This file contains distilled knowledge and context for AI assistants working on this project.

## Project Overview
This is a reinforcement learning research project studying **Train-Inference Mismatch** in LLM training using the RL2 framework.

## Research Goal
- Study how rollout-stage perturbations affect on-policy reinforcement learning
- Implement controlled perturbations to simulate real-world inference inconsistencies
- Validate findings on Qwen3 models using PPO algorithm

## Key Files
- **Main modification target**: `RL2/workers/rollout.py`
- **Configuration**: `configs/train/` (YAML files)
- **Examples**: `examples/` (reference scripts)

## Quick Start
```bash
# Install
pip install -e .

# Run baseline (example)
torchrun --nproc_per_node=1 -m RL2.trainer.ppo examples/orz_ppo.sh

# Run with custom config
torchrun --nproc_per_node=1 -m RL2.trainer.ppo <your_config.yaml>
```

## Vibe Coding Context
All AI interaction history and prompts are stored in `.vibe/` directory:
- `.vibe/prompts/`: Reusable prompt templates
- `.vibe/memory/`: Session records organized by date
- `.vibe/context/`: Long-term knowledge base

## Important Conventions
1. All modifications to RL2 core should be documented in `.vibe/memory/`
2. New prompt templates go in `.vibe/prompts/`
3. Research findings and decisions go in `.vibe/context/`
4. Experimental configs should be version-controlled in `configs/`
