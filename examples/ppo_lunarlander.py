import utils
utils.nowarnings()

import gym
from learn import PPO
import utils.torch, utils.env
import numpy as np
import torch
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from collections import deque
from baselines import bench

# Reaches MeanR = +100 in ~30k steps, 2 mins of training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_processes = 1
gamma = 0.99

env = gym.make("LunarLanderContinuous-v2")
env = utils.env.NormalizedActions(env)
env = utils.env.TimeLimit(env, 200)
env = utils.torch.make_deterministic(env)
env = bench.Monitor(env, filename = None)
envs = DummyVecEnv([lambda: env])
envs = utils.env.VecNormalize(envs, gamma = gamma)
envs = utils.env.VecPyTorch(envs, device)
utils.log('mc')

num_env_steps = int(1e6)
num_steps = 128
log_interval = 1
# tb = utils.torch.Summary('logs/lunarlander')
obs_space, action_space = envs.observation_space, envs.action_space
init_obs = envs.reset()

agent = PPO(
    obs_space,
    action_space,
    init_obs,
    num_steps = num_steps,
    num_processes = num_processes,
    gamma = gamma,
)

num_updates = agent.compute_updates_needed(num_env_steps, num_processes)
episode_rewards = deque(maxlen=10)
start = utils.timer()

for j in range(num_updates):

    agent.pre_step(j, num_updates)
    agent.step(envs, log = episode_rewards)
    vloss, piloss, ent = agent.train()

    if (j+1) % log_interval == 0 and len(episode_rewards) > 1:
        total_num_steps = (j + 1) * num_processes * num_steps
        elapsed = "Elapsed %s" % utils.timer_done(start)
        MeanR, MedR = np.mean(episode_rewards), np.median(episode_rewards)
        MinR, MaxR = np.min(episode_rewards), np.max(episode_rewards)
        reward_stats = "MeanR,MedR:%.2f,%.2f MinR,MaxR:%.2f,%.2f" % (MeanR, MedR, MinR, MaxR)
        loss_stats = "Ent:%f, VLoss:%f, PiLoss:%f" % (ent, vloss, piloss)
        stats = ["Steps:%g" % total_num_steps, elapsed, reward_stats, loss_stats]
        print(" ".join(stats))