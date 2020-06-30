"""
White-Box attacks on the Pong agent at inference time.

We use FGSM to compute adversarial perturbations for a trained Neural Network policy.

Here the adversary has access to the training environment, knowledge of the training algorithm and hyperparameters. It also knows the neural network architecture of the target policy and its parameters.

"""
import gym
from stable_baselines import TRPO, deepq, PPO2, logger
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.atari_wrappers import  make_atari, wrap_deepmind
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from gym.wrappers.monitoring import video_recorder
import torch
import torch.nn as nn 
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(precision = 8)
saved_wgts = 'ppo_pong'
attack_results = 'ppo_pong_attack_stats.txt'

ppo_model = PPO2.load(saved_wgts)

params = ppo_model.get_parameters()
param_list = ppo_model.get_parameter_list()

class network(nn.Module):
  # This network architecture is the same as the saved weights policy (White-Box attacks)
  def __init__(self, hin, win, actions):
    super(network, self).__init__()
    self.conv0 = nn.Conv2d(4, 32, kernel_size = 8, stride = 4, bias = True)
    self.conv1 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2, bias = True)
    self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, bias = True)
    def get_ops(size, kernel_size, stride):
            return (size - kernel_size)//stride + 1
    hout, wout = get_ops(get_ops(get_ops(hin, 8, 4), 4, 2), 3, 1), get_ops(get_ops(get_ops(win, 8, 4), 4, 2), 3, 1)
    self.linear = 64 * hout * wout
    self.lin0 = nn.Linear(self.linear, 512, bias = True)
    self.lin1 = nn.Linear(512, actions)

  def forward(self, t):
    t = f.relu(self.conv0(t))
    t = f.relu(self.conv1(t))
    t = f.relu(self.conv2(t))
    t = t.permute(0,2,3,1).reshape(-1)
    t = f.relu(self.lin0(t))
    t = self.lin1(t)
    # Output will be the logits
    return t

env = VecFrameStack(make_atari_env(env_id = 'PongNoFrameskip-v4', num_env = 1, seed = 3), 4)
env.reset().shape
h, w = env.reset().shape[1], env.reset().shape[2]
policy_net = network(h, w, env.action_space.n)

# Transfer of weights from saved model to newly created network
# conv weights
x = np.transpose(params['model/c1/w:0'], [3,2,0,1])
policy_net.conv0.weight = torch.nn.parameter.Parameter(torch.tensor(x))
x = np.transpose(params['model/c2/w:0'], [3,2,0,1])
policy_net.conv1.weight = torch.nn.parameter.Parameter(torch.tensor(x))
x = np.transpose(params['model/c3/w:0'], [3,2,0,1])
policy_net.conv2.weight = torch.nn.parameter.Parameter(torch.tensor(x))

# conv biases
x = params['model/c1/b:0'].reshape(params['model/c1/b:0'].shape[1])
policy_net.conv0.bias = torch.nn.parameter.Parameter(torch.tensor(x))
x = params['model/c2/b:0'].reshape(params['model/c2/b:0'].shape[1])
policy_net.conv1.bias = torch.nn.parameter.Parameter(torch.tensor(x))
x = params['model/c3/b:0'].reshape(params['model/c3/b:0'].shape[1])
policy_net.conv2.bias = torch.nn.parameter.Parameter(torch.tensor(x))

# lin weights
x = np.transpose(params['model/fc1/w:0'], [1,0])
policy_net.lin0.weight = torch.nn.parameter.Parameter(torch.tensor(x))

# lin biases
x = params['model/fc1/b:0']
policy_net.lin0.bias = torch.nn.parameter.Parameter(torch.tensor(x))

# pi wights
x = np.transpose(params['model/pi/w:0'], [1,0])
policy_net.lin1.weight = torch.nn.parameter.Parameter(torch.tensor(x))

# pi biases
x = params['model/pi/b:0']
policy_net.lin1.bias = torch.nn.parameter.Parameter(torch.tensor(x))

def fgsm_maxnorm_stacked(original_stack, epsilon, inp_grad_stack):
    # changing the pixels by epsilon in the direction of sign of gradient of loss function
    perturbed_stack = original_stack + epsilon * inp_grad_stack.sign()
    # clamp so that the pixels range between 0 and 1
    perturbed_stack = torch.clamp(perturbed_stack, 0, 1)
    return perturbed_stack

# PONG ACTION SPACE:
# [0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT', 4: 'RIGHTFIRE', 5: 'LEFTFIRE']

def generate_adv_example_stacked(epsilon):
    obs = env.reset()
    # proccessed observation
    pro_obs = np.transpose(obs, [0,3,1,2])/255.0
    pro_obs = torch.tensor(pro_obs, dtype = torch.float32, requires_grad = True)
    # list of rewards accumulated for the given epsilon on per epsiode basis
    ep_rew = [0.0]
    # number of episodes
    ep = 0
    # action without perturbation
    corr_action = 0
    # action with perturbation
    adv_action = 0
    for i in range(30000):
        # output logits
        logits = policy_net(pro_obs)
        out_prob = f.softmax(logits, dim = 0)
        # track the maximizing action as action before adding perturbation
        without_per_action = torch.tensor([out_prob.detach().argmax()])
        with_per_action = without_per_action.clone()

        # loss is the cross entropy loss between the output probability and the maximizing action without perturbation
        loss = f.cross_entropy(out_prob.reshape(1, out_prob.shape[0]), without_per_action)
        # zero the gradients
        if pro_obs.grad is not None:
            pro_obs.grad.data.zero_()
        # back prop
        loss.backward()
        # gradient wrt input
        inp_grad_stack = pro_obs.grad.data
        original_stack = pro_obs
        # stacked images after perturbation
        perturbed_stack = fgsm_maxnorm_stacked(original_stack, epsilon, inp_grad_stack)
        
        # Comparing the two stacked frames by plotting
        plt.figure(figsize = (20, 20))
        count = 1
        for i in range(4):
          plt.subplot(2, 4, count)
          plt.imshow(original_stack[0,i,:,:].detach().numpy())
          plt.subplot(2, 4, count + 4)
          plt.imshow(perturbed_stack[0,i,:,:].detach().numpy())
          count += 1

        # run forward pass to calculated the logits after perturbation
        per_logits = policy_net(perturbed_stack)
        per_out_prob = f.softmax(per_logits, dim = 0)
        # action taken after perturbation
        with_per_action = torch.tensor([per_out_prob.detach().argmax()])
        
        if with_per_action != without_per_action:
            # network picked different action from what it would pick without perturbation
            adv_action += 1
        else:
            # network picked the same initial action despite of perturbation
            corr_action += 1

        # take a step by performaing the perturbed action
        obs, reward, done, info = env.step(np.array(with_per_action))
        obs = np.transpose(obs, [0,3,1,2])/255.0
        pro_obs = torch.tensor(obs, dtype = torch.float32, requires_grad = True)
        
        ep_rew[-1] += reward
        env.render()
        vr.capture_frame()
        
        # episode completed
        if done:
            obs = env.reset()
            obs = np.transpose(obs, [0,3,1,2])/255.0
            pro_obs = torch.tensor(obs, dtype = torch.float32, requires_grad = True)
            
            string = 'Net reward for episode {} : {}'.format(ep, ep_rew[-1])
            print(string)
            f_ptr.write(string + '\n')
            if((ep+1)%5 == 0):
                string = 'Mean reward for the last 5 episodes: {}'.format(np.mean(ep_rew[-5:]))
                print(string)
                f_ptr.write(string + '\n')
            ep_rew.append(0.0)
            ep += 1
            string = 'Number of timesteps completed: {}'.format(i+1)
            print(string)
            f_ptr.write(string + '\n')
    env.close()
    vr.close()
    # pop the last episode reward to account for its possible incompleteness
    ep_rew.pop()
    return corr_action, adv_action, np.mean(ep_rew)

# Test the attack
epsilons = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
avg_return = []
f_ptr = open(attack_results, 'a')
for epsilon in epsilons:
  vr = video_recorder.VideoRecorder(env, base_path="./videos/Pong_test_after_attack_epsilon="+str(epsilon), enabled="./videos/Pong_test_after_attack_epsilon="+str(epsilon) is not None)
  string = 'Epsilon = {} \n'.format(epsilon)
  print(string)
  f_ptr.write(string)
  corr_action, adv_action, avg_rew = generate_adv_example_stacked(epsilon)
  avg_return.append(avg_rew)
  string = 'Epsilon = {}, Non-perturbed action percentage = {} \n'.format(epsilon, (corr_action/(corr_action + adv_action)) * 100)
  print(string)
  f_ptr.write(string)
f_ptr.flush()

# Plot results
def plot_result():
  plt.figure(figsize = (10, 10))
  plt.xlabel('EPSILON')
  plt.ylabel('AVERAGE RETURN')
  plt.plot(epsilons, avg_return)
  plt.grid()
  plt.show()

plot_result()