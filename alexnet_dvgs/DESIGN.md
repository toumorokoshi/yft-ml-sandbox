# Design

## Approach

1. after model training is complete, or with a pretrained model, load up the
   dataset that you would like to evaluate. Choose a random set from a subset of
   data.
   - run the same type of data conditioning as during training.
1. Initialize a list / map (precise data structure TBD) to store individual
   gradients per input.
   - run a batch size of 1 (allow us to capture the gradient of that particular data point).
   - In this specific case of AlexNet, capture the gradient for the 2nd to last
     layer. This is to not go all the way back to the original convolution
     layers which may target more general knowledge, and instead specific the
     layers related to the targetted model behavior and fine tuning.
1. Use cosine similarity to compare it to existing vectors in the list
   - If the similarity is less than some threshold (0.3) from any of the previous data points, add the new data point to the list.

The final output should be set of inputs, each of which is a representative
gradient which is sufficiently distinct from each other.

## Future optimizations

1. Reduce the dimensionality of that vector make it easier to compare gradients. (Johnson-Lindenstrauss projections)
