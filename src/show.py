import gym
from gym import Wrapper
from gym.wrappers.monitoring import video_recorder
from abstraction import Env
import numpy as np
import stable_baselines
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([lambda: Env()])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model = PPO2(MlpPolicy, env, verbose=1,learning_rate=0.003,nminibatches=4,n_steps=8000,ent_coef=0.0, gamma=0.95)
model = PPO2.load("knd3", env)

obs = env.reset()
rew=0
nn = 0
tt = 0
c1=0
c2=0
total=0
#video_recorder=None
#vr = video_recorder.VideoRecorder(env, "./videos/1.mp4", enabled="./videos/1.mp4" is not None)
for i in range(5000):
  action, _states = model.predict(obs)
  env.render()
  #vr.capture_frame()
  obs, rewards, done, info = env.step(action)
  rew+=rewards
  tt += rewards
  '''if done:
      if info[1] >= 900:
          c2+=1
      elif info[0] >= 900:
          c1+=1
      total+=1   
      print(tt)
      tt = 0'''

  if done:
      env.reset()
      #vr.close()
      #vr.enabled = False



print(c1,c2,c1+c2,total)
print("Total rewards",rew)
