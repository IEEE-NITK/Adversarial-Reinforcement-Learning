"""
Black-Box attacks on the Pong agent at inference time.

Here the adversary does not have compelete information about the agent. We use the Transferability property of adversarial examples in following two ways:

1. Transferability Across Policies: The adversary has access to the training environment and knowledge of the training algorithm and hyperparameters. It knows the neural network architecture of the target policy network, but not its random initialization.

2. Transferability Across Training Algorithms: The adversary additionally has no knowledge of the training algorithm or hyperparameters.

Following is the implementation of Transferability across algorithms

"""
