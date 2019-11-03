# Adapted from https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py
# Original code under MIT License, Copyright (c) 2017 Ilya Kostrikov

import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from utils.torch import cat_lists
import numpy as np

from .replay_buffer import ReplayMemory, Transition
from .networks import Actor, CriticTD3

class TD3(object):

    def __init__(self, num_inputs, action_space, lr = [3e-4, 3e-4], tau = [0.005, 0.005],
        gamma = 0.99, replay_size = 1000000, batch_size = 64, warmup_steps = 1000,
        action_noise = None, param_noise = None, policy_freq = 2, h = 256):

        self.num_inputs = num_inputs
        self.action_space = action_space
        self.actor_lr, self.critic_lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        h = [h, h]
        self.actor = Actor(self.actor_lr, self.num_inputs, self.action_space, h).to(self.device)
        self.actor_target = Actor(self.actor_lr, self.num_inputs, self.action_space, h).to(self.device)
        self.actor_perturbed = Actor(self.actor_lr, self.num_inputs, self.action_space, h).to(self.device)

        self.critic = CriticTD3(self.critic_lr, self.num_inputs, self.action_space, h).to(self.device)
        self.critic_target = CriticTD3(self.critic_lr, self.num_inputs, self.action_space, h).to(self.device)

        self.gamma = gamma
        self.tau_actor, self.tau_critic = tau
        self.memory = ReplayMemory(replay_size)
        self.batch_size = batch_size
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.warmup_steps = warmup_steps
        self.updates = 0
        self.policy_freq = policy_freq

        self.soft_update(self.actor_target, self.actor, tau = 1)
        self.soft_update(self.critic_target, self.critic, tau = 1)


    def select_action(self, state):
        self.actor.eval()
        if self.param_noise != None:
            mu = self.actor_perturbed(torch.Tensor(state).to(self.device))
        else:
            mu = self.actor(torch.Tensor(state).to(self.device))
        if self.action_noise != None:
            mu += torch.Tensor(self.action_noise()).to(self.device)
        self.actor.train()
        return mu.cpu().detach().numpy()

    def remember(self, state, action, done, next_state, reward):
        state, next_state = [state], [next_state]
        mask = [not done]
        reward = [reward]
        action = [action]
        self.memory.push(state, action, mask, next_state, reward)

    def update_parameters(self):

        if len(self.memory) < self.batch_size or \
            len(self.memory) < self.warmup_steps:
            return np.nan, np.nan

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = cat_lists(batch.state).to(self.device)
        action_batch = cat_lists(batch.action).to(self.device)
        reward_batch = cat_lists(batch.reward).to(self.device)
        mask_batch = cat_lists(batch.mask).to(self.device)
        next_state_batch = cat_lists(batch.next_state).to(self.device)
        
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

        next_action_batch = self.actor_target(next_state_batch)
        nsq1, nsq2 = self.critic_target(next_state_batch, next_action_batch)
        next_state_action_values = torch.min(nsq1, nsq2)
        cq1, cq2 = self.critic(state_batch, action_batch)

        targets = []
        for j in range(self.batch_size):
            targets.append(reward_batch[j] + self.gamma * mask_batch[j] * next_state_action_values[j])
        targets = torch.Tensor(targets).to(self.device)
        targets = targets.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        value_loss = torch.nn.SmoothL1Loss()(cq1, targets) + torch.nn.SmoothL1Loss()(cq2, targets)
        value_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()

        if self.updates % self.policy_freq == 0:

            self.actor.optimizer.zero_grad()
            self.actor.train()
            mu = self.actor(state_batch)
            policy_loss = -self.critic(state_batch, mu, both = False).mean()
            policy_loss.backward()
            self.actor.optimizer.step()
            self.actor.eval()

            self.soft_update(self.actor_target, self.actor, self.tau_actor)
            self.soft_update(self.critic_target, self.critic, self.tau_critic)

        return value_loss.item(), policy_loss.item()

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix) 
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def perturb_actor_parameters(self, param_noise):
        self.soft_update(self.actor_perturbed, self.actor, tau = 1)
        params = self.actor_perturbed.state_dict()
        for name in params:
            param = params[name]
            param += torch.randn(param.shape).to(self.device) * param_noise.current_stddev
    
    def adapt(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = cat_lists(batch.state).to(self.device)
        action_batch = cat_lists(batch.action).to(self.device)
        self.actor.eval()
        unperturbed_actions = self.actor(state_batch)
        self.actor.train()
        perturbed_actions = action_batch
        ddpg_dist = (perturbed_actions - unperturbed_actions)
        ddpg_dist = torch.sqrt(torch.mul(ddpg_dist, ddpg_dist).mean())
        ddpg_dist = ddpg_dist.cpu().detach().numpy()
        self.param_noise.adapt(ddpg_dist)