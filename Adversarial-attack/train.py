from stable_baselines import TRPO, deepq, PPO2, logger
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.atari_wrappers import  make_atari, wrap_deepmind
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from stable_baselines.common import set_global_seeds

def train_ppo(env_id, num_timesteps, seed, policy, n_envs=1, nminibatches=5, n_steps=8000):

    # env_id: typr str, identifies each environment uniquely
    # num_timesteps: number of timesteps to run the algorithm
    # seed: initial random seed
    # policy: policy to be followed (mlp, cnn, lstm, etc)
    # n_env: number of envs to run in parallel
    # nminibatches: number of minibatches of mini batch gradient descent (first-order optimization) to update the policy params
    # n_steps: number of steps in each update

    # Train PPO algorithm for num_timesteps
    # stack the frames for the vectorized environment
    # Note: PPO2 works only with vectorized environment
    env = VecFrameStack(make_atari_env(env_id, n_envs, seed), 4)
    # define the policy
    policy = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy, 'lnlstm': CnnLnLstmPolicy, 'mlp': MlpPolicy}[policy]
    # create model object for class PPO2
    model = PPO2(policy=policy, env=env, n_steps=n_steps, nminibatches=nminibatches, lam=0.95, gamma=0.99, noptepochs=4, ent_coef=.01,
                 learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1)
    # train the model
    # trained for 2e8 timesteps with seedo = 5
    model.learn(total_timesteps=num_timesteps)
    # save the hyperparameters and weights
    model.save('ppo'+env_id)
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
