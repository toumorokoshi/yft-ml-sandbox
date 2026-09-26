# Jepa RL Mario

## Example Commands

### Basic Training

```bash
bazel run //jepa_rl_mario:train -- --render-mode human --episodes 1 --steps 10000
```

### Save Final Checkpoint

```bash
bazel run //jepa_rl_mario:train -- --render-mode none --episodes 10 --save-checkpoint checkpoints/mario_dqn.pt
```

### Resume Training from Checkpoint

```bash
bazel run //jepa_rl_mario:train -- --render-mode none --load-checkpoint checkpoints/mario_dqn.pt --episodes 5 --save-checkpoint checkpoints/mario_dqn_updated.pt
```

### Evaluation Run from Checkpoint (Greedy Actions, No Updates)

```bash
bazel run //jepa_rl_mario:train -- --render-mode human --load-checkpoint checkpoints/mario_dqn.pt --eval --episodes 1 --steps 5000
```

## How does the reinforcement learning work?

Primarily in `run_episode` in `train.py`.

1. Steps run through an environment step.
2. `obs` is the numpy array with RGB pixels of the screen.
3. Preprocessed and reduced to grayscale via `preprocess_observation`.
4. `select_action` introduces random actions with probability `epsilon` to explore new actions, or takes greedy argmax Q-values.
5. Transitions stored in replay buffer; batches sampled to compute MSE Bellman loss and update `q_network`.

## Thoughts on the Mario Model

1. Grayscale input images
2. ViT for attention layers to extract feature embeddings with spatial awareness.
3. MLP decoder for action values.