%tensorflow_version 1.x
import os
import numpy as np
import matplotlib.pyplot as plt
import gym
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
from stable_baselines.common.callbacks import BaseCallback

log_dir = '/tmp/gym'
os.makedirs(log_dir, exist_ok=True)
save_params = 'dqn_pong_adv'
train_results = '/tmp/train_results.txt'
learning_starts = 1000

class save_best_model_callback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, save_freq, log_dir, save_params, verbose=0):
        super(save_best_model_callback, self).__init__(verbose)
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model_'+save_params)
        self.best_mean_reward = -np.inf

    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 50 episodes
              mean_reward = np.mean(y[-50:])
              if self.verbose > 0:
                callno = "Verbose called {} times \n\n".format(int((self.n_calls - learning_starts)/self.save_freq))
                timesteps = "Num timesteps: {} \n\n".format(self.num_timesteps)
                reward = "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f} \n\n".format(self.best_mean_reward, mean_reward)
                print(callno)
                print(timesteps)
                print(reward)
                with open(train_results, 'a') as f_ptr:
                  f_ptr.write(callno)
                  f_ptr.write(timesteps)
                  f_ptr.write(reward)

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    save_model = "Saving new best model to {} \n\n".format(self.save_path)
                    print(save_model)
                    with open(train_results, 'a') as f_ptr:
                      f_ptr.write(save_model)
                  self.model.save(self.save_path)

        return True

COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']
EPISODES_WINDOW = 100

def plot_results(dirs, num_timesteps, xaxis, task_name):
    tslist = []
    for folder in dirs:
        timesteps = load_results(folder)
        if num_timesteps is not None:
            timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        tslist.append(timesteps)
    xy_list = [ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]
    
    plt.figure(figsize=(8, 2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i]
        plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean, color=color)
    plt.xlim(minx, maxx)
    plt.title(task_name)
    plt.xlabel(xaxis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()

def train_dqn_adv(env_id, train_timesteps, seed, policy, save_params, n_envs = 1):
    set_global_seeds(seed)
    env = make_atari(env_id)
    env.seed(seed)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = wrap_deepmind(env, frame_stack=True)
    # define the policy
    policy = {'cnn': CnnPolicy, 'mlp': MlpPolicy}[policy]
    # create model object for class DQN
    model = DQN(policy = policy, env = env, gamma=0.99, learning_rate=0.0001, buffer_size=10000, exploration_fraction=0.1, exploration_final_eps=0.01, 
                exploration_initial_eps=1.0, train_freq=4, batch_size=32, double_q=True, learning_starts=10000, target_network_update_freq=1000, 
                prioritized_replay=True, prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-06, 
                param_noise=False, n_cpu_tf_sess=None, verbose=1)
    callback = save_best_model_callback(save_freq = 100, log_dir = log_dir, save_params = save_params, verbose=1)
    # train the model
    # trained for 2e7 timesteps with seed = 7
    model.learn(total_timesteps = train_timesteps, callback = callback)
    plot_results([log_dir], train_timesteps, results_plotter.X_TIMESTEPS, "DQNPong_TrainedByAdversary")
    plt.show()
    env.close()
    # free the memory
    del model