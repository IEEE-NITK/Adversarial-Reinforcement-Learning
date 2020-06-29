import gym
from abstraction import Env
import numpy as np
import stable_baselines
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([lambda: Env()])
env = VecNormalize(env, norm_obs=True, norm_reward=False)

model = PPO2(MlpPolicy, env, verbose=1,learning_rate=0.003,nminibatches=4,n_steps=8000,ent_coef=0.0, gamma=0.95)
model.learn(total_timesteps=10000000)
model.save("knd3")
obs = env.reset()
rew=0
nn = 0
tt = 0
for i in range(500):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  rew+=rewards
  tt += rewards
  if done:
      print(tt)
      tt = 0
  env.render()

print("Total rewards",rew)
