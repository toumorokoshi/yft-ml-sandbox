# Jepa RL Mario

## Example Command

`bazel run //jepa_rl_mario:train -- --render-mode human --episodes 1 --steps 10000`

<<<<<<< HEAD
## Thoughts on the Mario Model

1. grayscale input images
2. ViT for attention layers, this will extract feature embeddings. This should ensures that there are feature embeddings with the appropriate positions.
   1. check: does the ViT need positional encodings?
3. have an MLP decoder for actions?
4. still introduce random actions for decodings?
=======
## How does the reinforcement learning work?

Primarily in `run_episode` in train.py.

1. steps runs through a step
2. obs is the numpy array with the RGB of the image. This seves as the input.
3. preprocessed and reduced to grayscale.
4. select_action introduces some random actions to introduce some entropy, allowing it to learn new actions.
5. 5.
>>>>>>> 594c344f (feat: start on mario model)
