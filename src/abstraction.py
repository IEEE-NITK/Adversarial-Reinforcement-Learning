import gym
import gym_compete
from policy import LSTMPolicy, MlpPolicyValue
from gym import spaces
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary
import pickle
import sys
import argparse
import tensorflow as tf
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params

def setFromFlat(var_list, flat_params):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    tf.get_default_session().run(op, {theta: flat_params})

class Env():
  def __init__(self):
    self.metadata = {'render.modes': ['human']}
    super(Env, self).__init__()
    self.env = gym.make("multicomp/KickAndDefend-v0")
    policy_type = "lstm"
    self.rew1=0
    self.rew2=0
    self.now=0
    config = argparse.Namespace(env='kick-and-defend', max_episodes=1000000, param_paths=['agent-zoo/kick-and-defend/kicker/agent1_parameters-v1.pkl', 'agent-zoo/kick-and-defend/defender/agent2_parameters-v1.pkl'])

    param_paths = config.param_paths
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()

    self.policy = []
    self.policy.append(LSTMPolicy(scope="policy0", reuse=False,
                                     ob_space=self.env.observation_space.spaces[0],
                                     ac_space=self.env.action_space.spaces[0],
                                     hiddens=[128, 128], normalize=True))
    sess.run(tf.variables_initializer(tf.global_variables()))
    params = [load_from_file(param_pkl_path=path) for path in param_paths]

    setFromFlat(self.policy[0].get_variables(), params[0])

    self.action_space = spaces.Box(
            low=-1, high=1,shape=(17,))

        # Prices contains the OHCL values for the last five prices , dtype=np.float32
    self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,384))
    #self.reward_range = (0, 100)

  def reset(self):
    self.observation = self.env.reset()
    self.rew1=0
    self.rew2=0
    return self.observation[1]
  
  def step(self, action1):

    action0 = self.policy[0].act(stochastic=True, observation=self.observation[0])[0]
    action = (action0,np.asarray(action1))
    self.observation, reward, done, infos = self.env.step(action)
    self.rew1+=reward[0]
    self.rew2+=reward[1]
    self.now+=1
    if done:
        print("Agent 1",self.rew1,"Agent 2",self.rew2)


    return self.observation[1], reward[1]-reward[0], done, {}

  def render(self, mode='human', close=False):
    self.env.render()


#x = Env()
