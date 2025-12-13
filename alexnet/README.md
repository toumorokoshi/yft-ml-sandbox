# AlexNet

This is a (WIP) attempt to recreate the AlexNet paper.

## Results

### ImageNette

- with one linear layer (INPUT, 10): 9.9% accuracy.

### Scratch notes

### Model Arch 2x linear layer + relu

- layers of size 38416, then 512

| Accuracy | Epochs | Batch Size | Learning Rate | Shuffled | Momentum |
| -------- | ------ | ---------- | ------------- | -------- | -------- |
| 44.6%    | 10     | 32         | 0.01          | yes      | 0        |
| 42.9%    | 20     | 16         | 0.01          | yes      | 0        |
| 41.9%    | 10     | 16         | 0.001         | yes      | 0        |
| 40.9%    | 10     | 32         | 0.001         | yes      | 0        |
| 38.0%    | 5      | 16         | 0.001         | yes      | 0        |
| 11%      | 10     | 16         | 0.01          | no       | 0.9      |
| 9.9%     | 10     | 16         | 0.001         | no       | 0        |
| 0%       | 10     | 16         | 0.1           | yes      | 0        |