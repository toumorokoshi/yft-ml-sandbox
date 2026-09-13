# Jepa RL Mario

## Example Command

`bazel run //jepa_rl_mario:train -- --render-mode human --episodes 1 --steps 10000`

## Thoughts on the Mario Model

1. grayscale input images
2. ViT for attention layers, this will extract feature embeddings. This should ensures that there are feature embeddings with the appropriate positions.
   1. check: does the ViT need positional encodings?
3. have an MLP decoder for actions?
4. still introduce random actions for decodings?