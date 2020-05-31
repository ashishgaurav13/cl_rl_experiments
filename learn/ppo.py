import torch
import torch.nn as nn
import torch.optim as optim
from .networks import PolicyPPO
from .replay_buffer import RolloutStorage
import utils.torch
import numpy as np

class PPO():
    def __init__(self,
                 obs_space,
                 action_space,
                 init_obs,
                 clip_param = 0.1,
                 ppo_epoch = 4,
                 num_mini_batch = 32,
                 value_loss_coef = 0.5,
                 entropy_coef = 0,
                 lr = 3e-4,
                 eps = 1e-5,
                 max_grad_norm = 0.5,
                 use_clipped_value_loss = True,
                 num_steps = 128,
                 num_processes = None,
                 linear_schedule = True,
                 linear_schedule_mode = 0,
                 use_gae = True,
                 gae_lambda = 0.95,
                 use_proper_time_limits = False,
                 gamma = 0.99,
                 policy = None,
                 hidden = -1,
                 policy_class = PolicyPPO):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if policy != None:
            print("Loading provided policy!")
            self.actor_critic = policy
        else:
            if hidden == -1: hidden = 64
            print("Creating networks with hidden = %d" % hidden)
            print("policy_class = %s" % policy_class.__name__)
            self.actor_critic = policy_class(obs_space.shape, action_space,
                hidden_size = hidden).to(self.device)

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.num_steps = num_steps

        assert(num_processes != None)
        self.num_processes = num_processes
        self.rollouts = RolloutStorage(
            self.num_steps,
            self.num_processes,
            obs_space.shape,
            action_space
        )
        self.rollouts.obs[0].copy_(init_obs)
        self.rollouts.to(self.device)

        self.lr = lr
        self.linear_schedule = linear_schedule
        self.linear_schedule_mode = linear_schedule_mode
        actor_params = []
        critic_params = []
        for k, v in self.actor_critic.named_parameters():
            if "critic" in k:
                critic_params += [v]
            else:
                actor_params += [v]
        self.actor_optimizer = optim.RMSprop(actor_params, lr=lr, eps = eps)
        self.critic_optimizer = optim.RMSprop(critic_params, lr=lr, eps = eps)
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits
        self.gamma = gamma

    def compute_updates_needed(self, steps, num_processes):
        return steps // self.num_steps // num_processes

    def pre_step(self, j, num_updates):
        if self.linear_schedule:
            # decrease learning rate linearly
            utils.torch.update_linear_schedule(self.actor_optimizer, j, num_updates, self.lr, mode = self.linear_schedule_mode)
            utils.torch.update_linear_schedule(self.critic_optimizer, j, num_updates, self.lr, mode = self.linear_schedule_mode)

    def step(self, envs, log = None):

        for step in range(self.num_steps):

            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = self.actor_critic.act(
                    self.rollouts.obs[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            if log != None:
                assert(type(log) == dict)
                if 'traj' in log.keys():
                    log['traj'].see(obs, reward, done, infos)
                for info in infos:
                    if 'episode' in info.keys():
                        log['r'].append(info['episode']['r'])
                        log['eps_done'] += 1
                        if 'satisfactions' in info.keys():
                            log['satisfactions'] += info['satisfactions']
                        if len(done) == 1:
                            envs.reset() # Only works for one env (TODO)
                    
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
            self.rollouts.insert(obs, action, action_log_prob, value,
                reward, masks, bad_masks)

    def train(self):
        with torch.no_grad():
            next_value = self.actor_critic.get_value(self.rollouts.obs[-1]).detach()

        self.rollouts.compute_returns(next_value, self.use_gae, self.gamma,
                                    self.gae_lambda, self.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = self.update(self.rollouts)
        self.rollouts.after_update()
        return value_loss, action_loss, dist_entropy

    def compute_regularization_loss(self):
        return torch.tensor(0.0).to(self.device)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        reg_loss_epoch = 0

        for e in range(self.ppo_epoch):

            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                reg_loss = self.compute_regularization_loss()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                (value_loss * self.value_loss_coef).backward()
                (action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                if reg_loss != torch.tensor(0.0).to(self.device):
                    self.actor_optimizer.zero_grad()
                    reg_loss.backward()
                    self.actor_optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                reg_loss_epoch += reg_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        reg_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, [dist_entropy_epoch, reg_loss_epoch]