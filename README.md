# Adversarial Attacks and Defenses in Reinforcement Learning

The aim of this project was to explore Adversarial Attacks and Defenses in Single as well as Multi-Agent Reinforcement Learning. In the Single-Agent domains, we focus on Pixel-Based attacks in [Atari](https://gym.openai.com/envs/#atari) games from the [Gym](https://gym.openai.com/) environments. In Multi-Agent, we concentrate on attacking by training Adversarial Policies in 1-vs-1 zero-sum continuous control robotic environments from the [MuJoCo](http://www.mujoco.org/) simulator. We also studied potential defense procedures to counter such attacks.

A detailed article about the methods and approaches studied during the project can be found [here](https://aarl-ieee-nitk.github.io/reinforcement-learning,/adversarial/attacks,/defense/mechanisms/2020/04/09/Survey-on-Adversarial-attacks-and-defenses.html). We have also implemented some of these in this repository.

We also have a [blog](https://aarl-ieee-nitk.github.io/) with articles on the several concepts involved in the project.

## Structure

* `LearningPhaseAssignments` contains the Reinforcement Learning algorithms implemented during the learning phase of the project. This includes:
  * Tabular [SARSA](http://incompleteideas.net/book/first/ebook/node64.html) & [Q-Learning](http://incompleteideas.net/book/first/ebook/node65.html)
  * [Deep Q-Networks](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (DQN)
  * [Vanilla Policy Gradients](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) (VPG/[REINFORCE](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf))
  
* [`Adversarial-policies`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/tree/master/Adversarial-policies) contains a Tensorflow implementation of the attack by training Adversarial policies.

  [Gleave et al., 2020](https://arxiv.org/abs/1905.10615)
  
  The implementation in this folder is structured as follows:
    * [`agent-zoo`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/tree/master/Adversarial-policies/agent-zoo): Contains the pre-trained agent parameters for the environments described in [Bansal et al., 2018a](https://arxiv.org/abs/1710.03748). [Source](https://github.com/openai/multiagent-competition)
    * [`abstraction.py`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/blob/master/Adversarial-policies/abstraction.py): A wrapper over the Multi-Agent environment (Two player Markov game) to use it as Single-Agent. 
    * [`policy.py`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/blob/master/Adversarial-policies/policy.py): Contains the implementation of MLP and LSTM network policies of the agents.
    * [`train.py`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/blob/master/Adversarial-policies/train.py): Contains the code for training the adversarial policy using [Proximal Policy Optmization](https://aarl-ieee-nitk.github.io/reinforcement-learning,/policy-gradient-methods,/sampled-learning,/optimization/theory/2020/03/25/Proximal-Policy-Optimization.html) (PPO).
    * [`show.py`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/blob/master/Adversarial-policies/show.py): Contains the testing and video-making part.
    * [`finallog.txt`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/blob/master/Adversarial-policies/finallog.txt): Output logs from the training procedure.
    * [`knd3.zip`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/blob/master/Adversarial-policies/knd3.zip): Trained parameters for the adversarial policy in the Kick-and-Defend environment.
    * [`videos`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/tree/master/Adversarial-policies/videos/adversarial): Video displaying the adversarial attack in Kick-and-Defend.

* [`FGSM-on-Images`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/tree/master/FGSM-on-images) contains a PyTorch implementation of Pixel-based attacks on images and output plots and images with varying perturbations.

  * [`fast_gradient_sign_method.py`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/blob/master/FGSM-on-images/fast_gradient_sign_method.py): Contains the implementation of the [Fast Gradient Sign Method](https://arxiv.org/abs/1412.6572) (FGSM) on the MNIST dataset.
  
* [`Adversarial-attacks-on-DNN-policies`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/tree/master/Adversarial-attacks-on-DNN-policies): Contains a PyTorch implementation of the FGSM attack on Neural Network policies in Atari Pong environment.

  [Huang et al., 2017](https://arxiv.org/abs/1702.02284)

  * [`Adversarial-Attack`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/tree/master/Adversarial-attacks-on-DNN-policies/Adversarial-Attack): Contains the code, stats, and videos for adversarial attack on the Pong agent in [`WhiteBox`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/tree/master/Adversarial-attacks-on-DNN-policies/Adversarial-Attack/WhiteBox-attacks) as well as [`BlackBox`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/tree/master/Adversarial-attacks-on-DNN-policies/Adversarial-Attack/BlackBox-attacks) conditions.
  * [`Test`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/tree/master/Adversarial-attacks-on-DNN-policies/Test): Code, stats, and videos for the Pong agent before the adversarial attack.
  * [`Train`](https://github.com/IEEE-NITK/Adversarial-Reinforcement-Learning/tree/master/Adversarial-attacks-on-DNN-policies/Train): Code and videos for training a Pong agent using [PPO](https://aarl-ieee-nitk.github.io/reinforcement-learning,/policy-gradient-methods,/sampled-learning,/optimization/theory/2020/03/25/Proximal-Policy-Optimization.html).
  
## Requirements
* [PyTorch](https://pytorch.org/) (for Pixel attacks)
* [Tensorflow](https://www.tensorflow.org/) (for Adversarial policies)
* [Stable-Baselines](https://github.com/hill-a/stable-baselines) (2.9.0)
* [MuJoCo](http://www.mujoco.org/) 131

## Team
* Madhuparna Bhowmik
* Akash Nair
* Saurabh Agarwala
* Videh Raj Nema
* Kinshuk Kashyap
* Manav Singhal

Mentor: Moksh Jain