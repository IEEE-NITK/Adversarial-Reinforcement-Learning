from stable_baselines import TRPO, deepq, PPO2, logger
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.atari_wrappers import  make_atari, wrap_deepmind
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
import numpy as np
import matplotlib.pyplot as plt

def test_ppo(env_id, seed, n_envs = 1, path_to_policy_params):
    
    """
     env_id: typr str, identifies each environment uniquely
     num_timesteps: number of timesteps to run the algorithm
     seed: initial random seed
     policy: policy to be followed (mlp, cnn, lstm, etc)
     n_env: number of envs to run in parallel
     nminibatches: number of minibatches of mini batch gradient descent (first-order optimization) to update the policy params
     n_steps: number of steps in each update
    """
    
    # Train PPO algorithm for num_timesteps
    # stack 4 frames for the vectorized environment
    # Note: PPO2 works only with vectorized environment
    env = VecFrameStack(make_atari_env(env_id = env_id, num_env = n_envs, seed=seed), 4)
    # define the policy
    # create model object for class PPO2
    # The policy is CnnPolicy from stable baselines and has been trained for 2e7 time steps on Pong
    model = PPO2.load(path_to_policy_params)
    
    obs = env.reset()
    ep_rew = [0.0]
    ep = 0
    for i in range(50000):
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      ep_rew[-1] += rewards
      if dones:
        obs = env.reset()
        print('Net reward for episode ',ep,': ',ep_rew[-1])
        if((ep+1)%10 == 0):
          print('Mean reward for last 10 episodes: ',np.mean(ep_rew[-10:]))
        ep_rew.append(0.0)
        ep += 1
        print('Number of timesteps completed: ', i+1)

def test_trpo(env_id, seed, n_envs = 1):
  pass


def test_dqn(env_id, seed, n_envs = 1):
  pass


env.close()