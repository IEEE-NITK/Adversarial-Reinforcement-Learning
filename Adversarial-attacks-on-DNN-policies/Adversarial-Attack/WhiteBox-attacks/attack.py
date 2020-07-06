"""
White-Box attacks on the Pong agent at inference time.

We use FGSM to compute adversarial perturbations for a trained Neural Network policy. 
Implemented L1, L2, Linf attacks. Two versions of L1 attacks have been implemented.
Policies used - Two different PPO policies and one DQN policy

Here the adversary has access to the training environment, knowledge of the training algorithm and hyperparameters. It also knows the neural network architecture of the target policy and its parameters.

"""
%tensorflow_version 1.x
# use pip install git+https://github.com/hill-a/stable-baselines for stable baselines version 2.10.1a1
import stable_baselines
print(stable_baselines.__version__)
import os
import gym
from stable_baselines import TRPO, DQN, deepq, PPO2, logger
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
import pickle
import bottleneck

def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params

torch.set_printoptions(precision = 8)

# default file locations and names
# change the filenames here 

attack_results = '/ppoAdv_ppoVic_linf_attack_results.txt'

path_to_dqnparams = '/dqn/PongNoFrameskip-v4.pkl'
dqn_params = load_from_file(path_to_dqnparams)

path_to_ppoparams = '/ppo/PongNoFrameskip-v4.pkl'
ppo_params = load_from_file(path_to_ppoparams)

path_to_ppo2params = '/ppo2_pong'
ppo2_model = PPO2.load(path_to_ppo2params)
ppo2_params = ppo2_model.get_parameters()
ppo2_param_list = ppo2_model.get_parameter_list()

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

env = VecFrameStack(make_atari_env(env_id = 'PongNoFrameskip-v4', num_env = 1, seed = 2), 4)
env.reset().shape
h, w = env.reset().shape[1], env.reset().shape[2]
dqn_net = network(h, w, env.action_space.n)
ppo_net = network(h, w, env.action_space.n)
ppo2_net = network(h, w, env.action_space.n)

# params ppo
# Transfer of weights from saved model to newly created network
# conv weights
x = np.transpose(ppo_params[1][0], [3,2,0,1])
ppo_net.conv0.weight = torch.nn.parameter.Parameter(torch.tensor(x))
x = np.transpose(ppo_params[1][2], [3,2,0,1])
ppo_net.conv1.weight = torch.nn.parameter.Parameter(torch.tensor(x))
x = np.transpose(ppo_params[1][4], [3,2,0,1])
ppo_net.conv2.weight = torch.nn.parameter.Parameter(torch.tensor(x))

# conv biases
x = ppo_params[1][1].reshape(ppo_params[1][1].shape[1])
ppo_net.conv0.bias = torch.nn.parameter.Parameter(torch.tensor(x))
x = ppo_params[1][3].reshape(ppo_params[1][3].shape[1])
ppo_net.conv1.bias = torch.nn.parameter.Parameter(torch.tensor(x))
x = ppo_params[1][5].reshape(ppo_params[1][5].shape[1])
ppo_net.conv2.bias = torch.nn.parameter.Parameter(torch.tensor(x))

# lin weights
x = np.transpose(ppo_params[1][6], [1,0])
ppo_net.lin0.weight = torch.nn.parameter.Parameter(torch.tensor(x))

# lin biases
x = ppo_params[1][7]
ppo_net.lin0.bias = torch.nn.parameter.Parameter(torch.tensor(x))

# pi wights
x = np.transpose(ppo_params[1][10], [1,0])
ppo_net.lin1.weight = torch.nn.parameter.Parameter(torch.tensor(x))

# pi biases
x = ppo_params[1][11]
ppo_net.lin1.bias = torch.nn.parameter.Parameter(torch.tensor(x))

# params ppo another (ppo2)
# Transfer of weights from saved model to newly created network
# conv weights
x = np.transpose(ppo2_params['model/c1/w:0'], [3,2,0,1])
ppo2_net.conv0.weight = torch.nn.parameter.Parameter(torch.tensor(x))
x = np.transpose(ppo2_params['model/c2/w:0'], [3,2,0,1])
ppo2_net.conv1.weight = torch.nn.parameter.Parameter(torch.tensor(x))
x = np.transpose(ppo2_params['model/c3/w:0'], [3,2,0,1])
ppo2_net.conv2.weight = torch.nn.parameter.Parameter(torch.tensor(x))

# conv biases
x = ppo2_params['model/c1/b:0'].reshape(ppo2_params['model/c1/b:0'].shape[1])
ppo2_net.conv0.bias = torch.nn.parameter.Parameter(torch.tensor(x))
x = ppo2_params['model/c2/b:0'].reshape(ppo2_params['model/c2/b:0'].shape[1])
ppo2_net.conv1.bias = torch.nn.parameter.Parameter(torch.tensor(x))
x = ppo2_params['model/c3/b:0'].reshape(ppo2_params['model/c3/b:0'].shape[1])
ppo2_net.conv2.bias = torch.nn.parameter.Parameter(torch.tensor(x))

# lin weights
x = np.transpose(ppo2_params['model/fc1/w:0'], [1,0])
ppo2_net.lin0.weight = torch.nn.parameter.Parameter(torch.tensor(x))

# lin biases
x = ppo2_params['model/fc1/b:0']
ppo2_net.lin0.bias = torch.nn.parameter.Parameter(torch.tensor(x))

# pi wights
x = np.transpose(ppo2_params['model/pi/w:0'], [1,0])
ppo2_net.lin1.weight = torch.nn.parameter.Parameter(torch.tensor(x))

# pi biases
x = ppo2_params['model/pi/b:0']
ppo2_net.lin1.bias = torch.nn.parameter.Parameter(torch.tensor(x))

# params dqn
# Transfer of weights from saved model to newly created network
# conv weights
x = np.transpose(dqn_params[1][1], [3,2,0,1])
dqn_net.conv0.weight = torch.nn.parameter.Parameter(torch.tensor(x))
x = np.transpose(dqn_params[1][3], [3,2,0,1])
dqn_net.conv1.weight = torch.nn.parameter.Parameter(torch.tensor(x))
x = np.transpose(dqn_params[1][5], [3,2,0,1])
dqn_net.conv2.weight = torch.nn.parameter.Parameter(torch.tensor(x))

# conv biases
x = dqn_params[1][2].reshape(dqn_params[1][2].shape[1])
dqn_net.conv0.bias = torch.nn.parameter.Parameter(torch.tensor(x))
x = dqn_params[1][4].reshape(dqn_params[1][4].shape[1])
dqn_net.conv1.bias = torch.nn.parameter.Parameter(torch.tensor(x))
x = dqn_params[1][6].reshape(dqn_params[1][6].shape[1])
dqn_net.conv2.bias = torch.nn.parameter.Parameter(torch.tensor(x))

# lin weights
x = np.transpose(dqn_params[1][7], [1,0])
dqn_net.lin0.weight = torch.nn.parameter.Parameter(torch.tensor(x))

# lin biases
x = dqn_params[1][8]
dqn_net.lin0.bias = torch.nn.parameter.Parameter(torch.tensor(x))

# pi wights
x = np.transpose(dqn_params[1][9], [1,0])
dqn_net.lin1.weight = torch.nn.parameter.Parameter(torch.tensor(x))

# pi biases
x = dqn_params[1][10]
dqn_net.lin1.bias = torch.nn.parameter.Parameter(torch.tensor(x))

############################################## ATTACKS ##############################################

def fgsm_linfnorm_stacked(original_stack, epsilon, inp_grad_stack):
    # changing the pixels by epsilon in the direction of sign of gradient of loss function
    perturbation = epsilon * inp_grad_stack.sign()
    perturbed_stack = original_stack + perturbation
    # clamp so that the pixels range between 0 and 1
    perturbed_stack = torch.clamp(perturbed_stack, 0, 1)
    return perturbed_stack, perturbation

# to avoid division by 0
avoid_zero_div = torch.tensor([1e-12])
def fgsm_l2norm_stacked(original_stack, epsilon, inp_grad_stack):
    # number of dimensions of the inpur
    d = 1
    for i in original_stack.shape:
        d *= i
    # computing the perturbation
    eta = torch.tensor([epsilon * np.sqrt(d)])
    l2norm_grad = torch.sqrt(torch.max(torch.sum(inp_grad_stack**2), avoid_zero_div))
    per = inp_grad_stack/l2norm_grad
    perturbation = torch.mul(eta, per)
    # ensure that perturbation is within epsilon limits
    perturbation = torch.clamp(perturbation, -epsilon, epsilon)
    # perturbed stack
    perturbed_stack = original_stack + perturbation
    # clamping to ensure the pixel limits
    perturbed_stack = torch.clamp(perturbed_stack, 0, 1)
    return perturbed_stack.float(), perturbation.float()

# images = (1 ,4, 84, 84) in this case
n_frames = 4

# this attack modifies only the highest intensity pixels in each frame by maximal amount
def fgsm_l1norm_stacked(original_stack, epsilon, inp_grad_stack):
    # number of dimensions of the input
    d = 1
    for i in original_stack.shape:
        d *= i
    # sign matrix for gradients
    sign = inp_grad_stack.sign()
    # gradient matrix with absolute values
    abs_inp_grad_stack = torch.abs(inp_grad_stack)
    # pixels with highest intensity in each frame
    max_inp_grad_each_frame = torch.max(torch.max(abs_inp_grad_stack, dim=3).values, dim=-1).values.reshape(1,n_frames,1,1)
    # stack with pixels with the values computed above
    max_mask = torch.zeros(inp_grad_stack.shape) + max_inp_grad_each_frame
    # stack with 1 at the indices where highest pixel intensities exist and 0 at rest
    bool_mask = (max_mask == abs_inp_grad_stack).float()
    # perturbation and bringing the sign back
    perturbation = bool_mask * sign * epsilon * d
    # perturbed stack
    perturbed_stack = original_stack + perturbation
    # clamping to ensure the pixel limits
    perturbed_stack = torch.clamp(perturbed_stack, 0, 1)
    return perturbed_stack, perturbation

# this attack perturbs certain number of pixels in each frame by maximal amount
n_pixels = 6 # number of pixels to maximally perturb per frame
mat = h*w
def fgsm_l1norm_stacked_stronger(original_stack, epsilon, inp_grad_stack):
    d = 1
    for i in original_stack.shape:
        d *= i
    # sign matrix for gradients
    sign = inp_grad_stack.sign()
    # gradient matrix with absolute values
    abs_inp_grad_stack = torch.abs(inp_grad_stack)
    # perturbation and perturbed stacks
    perturbation = torch.zeros(1,n_frames,h,w)
    perturbed_stack = torch.zeros(1,n_frames,h,w)
    
    for i in range(n_frames):
        # unrolling into 1-D tensor
        frame_flat = abs_inp_grad_stack[:,i,:,:].reshape(mat)
        # pixels with maximal intensities per frame (number of pixels controlled by n_pixles)
        pixel_inds = torch.topk(input = frame_flat, k = n_pixels).indices
        # 1s at the indices of maximal values
        bool_mask = torch.zeros(mat)
        bool_mask[pixel_inds] = 1
        # perturbation_budget = epsilon * d
        perturbation[:,i,:,;] = bool_mask.reshape(h, w) * sign[:,i,:,:] * epsilon * d
        perturbed_stack[:,i,:,:] = original_stack[:,i,:,:] + perturbation[:,i,:,;]

    perturbed_stack = torch.clamp(perturbed_stack, 0, 1)
    return perturbed_stack, perturbation

#####################################################################################################

# to use different policies, algorithms and attacks
net_dict = {'ppo': ppo_net, 'dqn': dqn_net, 'ppo2': ppo2_net}
norm_dict = {'l1': fgsm_l1norm_stacked, 'l2': fgsm_l2norm_stacked, 'linf': fgsm_linfnorm_stacked, 'l1stronger': fgsm_l1norm_stacked_stronger}

# PONG ACTION SPACE:
# [0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT', 4: 'RIGHTFIRE', 5: 'LEFTFIRE']

# white box attack
# RESULTS FOR L1 ATTACK FOR L1 NORM WHITE-BOX ARE FOR HIGHEST PIXEL PERTURBATION - fgsm_l1norm_stacked
def generate_adv_example_stacked_white(epsilon = 0.0, victim = 'ppo', norm = 'linf', timesteps = 5000):
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
    for i in range(timesteps):
        # output of neural net
        logits = net_dict[victim](pro_obs)
        out_prob = f.softmax(logits, dim = 0)
        
        # for policy-based algorithms
        if victim == 'ppo' or victim == 'ppo2':
            action_probs = out_prob.detach().numpy()
            # sampling according to the output probability distribution
            action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
            without_per_action = torch.tensor([action])
        
        # for value-based algorithms
        elif victim == 'dqn':
            # track the maximizing action as action before adding perturbation
            without_per_action = torch.tensor([logits.detach().argmax()])

        with_per_action = without_per_action.clone()

        # loss is the cross entropy loss between the output probability/Q-values after softmax and the maximizing action without perturbation
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
        perturbed_stack, perturbation = norm_dict[norm](original_stack, epsilon, inp_grad_stack)
        
        # Comparing the two stacked frames by plotting
        plt.figure(figsize = (20, 20))
        count = 1
        for i in range(n_frames):
          plt.subplot(3, n_frames, count)
          plt.imshow(original_stack[0,i,:,:].detach().numpy())
          plt.subplot(3, n_frames, count + n_frames)
          plt.imshow(perturbation[0,i,:,:].detach().numpy())
          plt.subplot(3, n_frames, count + n_frames * 2)
          plt.imshow(perturbed_stack[0,i,:,:].detach().numpy())
          count += 1
        
        # output of the neural net after perturbation
        per_logits = net_dict[victim](perturbed_stack)
        
        # for policy-based algorithms
        if victim == 'ppo' or victim == 'ppo2':
            per_out_prob = f.softmax(per_logits, dim = 0)
            action_probs = per_out_prob.detach().numpy()
            # sampling according to the output probability distribution
            action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
            with_per_action = torch.tensor([action])
        
        # for value-based algorithms
        elif victim == 'dqn':
            # track the maximizing action as action before adding perturbation
            with_per_action = torch.tensor([per_logits.detach().argmax()])

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
        #vr.capture_frame()
        
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
    #vr.close()
    # pop the last episode reward to account for its possible incompleteness
    ep_rew.pop()
    return corr_action, adv_action, np.mean(ep_rew)

# Test the attack
env.reset()
epsilons = [0.0, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.03, 0.05]
avg_return = []
f_ptr = open(attack_results, 'a')
for epsilon in epsilons:
  env.reset()
  #vr = video_recorder.VideoRecorder(env, base_path="./videos/Pong_test_after_attack_epsilon="+str(epsilon), enabled="./videos/Pong_test_after_attack_epsilon="+str(epsilon) is not None)
  string = 'Epsilon = {} \n'.format(epsilon)
  print(string)
  f_ptr.write(string)
  corr_action, adv_action, avg_rew = generate_adv_example_stacked_white(epsilon = epsilon, victim = 'dqn', norm = 'l1', timesteps = 20000)
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