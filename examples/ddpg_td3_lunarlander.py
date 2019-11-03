import utils
utils.nowarnings()

import gym
from learn import TD3, OUNoise, AdaptiveParamNoiseSpec
import utils.torch, utils.env
import numpy as np
import torch

env = gym.make("LunarLanderContinuous-v2")
env = utils.env.NormalizedActions(env)
env = utils.env.TimeLimit(env, 200)
env = utils.torch.make_deterministic(env)
utils.log('mc')

num_episodes = 5000
# tb = utils.torch.Summary('logs/lunarlander')
n_obs, action_space = env.observation_space.shape[0], env.action_space
num_actions = action_space.shape[0]
action_noise = OUNoise(mu = 0.0 * np.ones(num_actions))
param_noise = None # AdaptiveParamNoiseSpec(0.05, 0.3, 1.05)
T = 1000
agent = TD3(n_obs, action_space, warmup_steps = T, policy_freq = 3, lr = [3e-5, 3e-6],
    action_noise = action_noise, param_noise = param_noise)

policy = lambda obs: agent.select_action(obs)
process_experience = lambda s, a, r, n, d: agent.remember(s, a, d, n, r)
train = lambda: agent.update_parameters()

rewards = []
total_numsteps = 0

begin_time = start_time = utils.timer()
for i_episode in range(num_episodes):

    # agent.perturb_actor_parameters(param_noise)
    T, R = utils.env.episode(env, policy, process_experience,
        train, episode_num = i_episode+1, history = rewards)
    # agent.adapt()


    if (i_episode+1) % 100 == 0:
        print('Last 100 epochs: %s' % utils.timer_done(start_time))
        start_time = utils.timer()

print('Overall: %s' % utils.timer_done(begin_time))
env.close()