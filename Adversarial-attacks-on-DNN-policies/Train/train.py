import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO, deepq, PPO2, logger
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN

log_dir = '/tmp/gym'
os.makedirs(log_dir, exist_ok=True)
save_params = 'ppo_pong'
best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step
    Params:
        _locals: (dict)
        _globals: (dict)
    """
    global n_steps, best_mean_reward
    if (n_steps+1)%10 == 0:
      print('Saving the Model.. (every 10 updates)')
      _locals['self'].save('ppo_pong')
    n_steps += 1
    print('n_steps = ', n_steps)

    x, y = ts2xy(load_results(log_dir), 'timesteps')
    if len(x) > 0:
    # Mean reward over last 10 episodes
      if(len(x) % 10 == 0):
        mean_reward = np.mean(y[-10:])
        print("Best mean reward over last 10 episodes: {:.2f} - Mean reward over last 10 episodes: {:.2f}".format(best_mean_reward, mean_reward))
        if mean_reward > best_mean_reward:
          best_mean_reward = mean_reward
      # Timesteps passed
      print(x[-1], 'timesteps')
    # Returning False will stop training early
    return True


def train_ppo(env_id, num_timesteps, seed, policy, save_params, n_envs = 1, nminibatches = 5, n_steps = 8000):

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
    # stack the frames for the vectorized environment
    # Note: PPO2 works only with vectorized environment

    set_global_seeds(seed)
    env = make_atari(env_id)
    env.seed(seed)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = wrap_deepmind(env, frame_stack=True)
    # define the policy
    policy = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy, 'lnlstm': CnnLnLstmPolicy, 'mlp': MlpPolicy}[policy]
    # create model object for class PPO2
    model = PPO2(policy = policy, env = env, n_steps = n_steps, nminibatches = nminibatches, lam = 0.95, gamma = 0.99, 
                    noptepochs = 4, ent_coef = .01, learning_rate = lambda f: f * 2.5e-4, cliprange = lambda f: f * 0.1, verbose = 1)
    # train the model
    # trained for 2e7 timesteps with seed = 5
    model.learn(total_timesteps = num_timesteps, callback = callback)
    # save the hyperparameters and weights
    model.save(save_params)
    env.close()
    # free the memory
    del model


def train_trpo(env_id, num_timesteps, seed):

    # env_id: typr str, identifies each environment uniquely
    # num_timesteps: number of timesteps to run the algorithm
    # seed: initial random seed

    # set up the environment
    rank = MPI.COMM_WORLD.Get_rank()
    sseed = seed + 10000 * rank
    set_global_seeds(sseed)
    env = make_atari(env_id)
    env.seed(sseed)
    env = wrap_deepmind(make_atari(env_id))
    env.seed(sseed)
    # define policies
    policy = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy, 'lnlstm': CnnLnLstmPolicy, 'mlp': MlpPolicy}[policy]
    # define TRPO class object
    model = TRPO(policy=policy, env=env, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_dampling=1e-3, ent_coef=0.0, gamma=0.99,
                 lam=1, vf_iters=3, vf_stepsize=1e-4, verbose=1)
    # Train TRPO for num_timesteps
    model.learn(total_timesteps=num_timesteps)
    # save the hyperparameters and weights
    model.save('trpo'+env_id)
    env.close()
    # free the memory
    del model

#def train_dqn():


#def train_a2c():
