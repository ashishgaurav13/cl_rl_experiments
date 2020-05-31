from .ppo import PPO
from .networks import PolicyPPO
import torch
import tqdm
import numpy as np

class PolicyPPO_EWC(PolicyPPO):

    def snapshot(self, trajectories, trajnum = 100, multiplier = 1, step_every = 25):
        if not hasattr(self, "ppi"):
            self.ppi = {}
            self.snapshots = {}
            self.multiplier = multiplier
            self.step_every = step_every
        if not hasattr(self, "tasknum"):
            self.tasknum = 0
        else:
            self.tasknum += 1
        tasknum = self.tasknum
        assert(tasknum not in self.ppi.keys())
        self.ppi[tasknum] = {}
        self.snapshots[tasknum] = {}
        for pname, param in self.named_parameters():
            if "critic" not in pname:
                with torch.no_grad():
                    self.ppi[tasknum][pname] = torch.zeros_like(param)
        count = 0
        relevant_trajectories = np.random.permutation(len(trajectories))[:trajnum]
        for ti in tqdm.tqdm(relevant_trajectories):
            t = trajectories[ti]
            for s in t:
                count += 1
                self.zero_grad()
                _, actor_features = self.forward(s)
                action_mode = self.dist.linear(actor_features)
                action_log_ll = torch.max(torch.nn.LogSoftmax()(action_mode))
                # The probability of action_mode for a normal distribution doesn not change
                # with the input. However, the action_mode itself changes. Hence we compute
                # the per-parameter importance using the gradients of action_mode with respect
                # to the input.
                action_log_ll.backward()
                for pname, param in self.named_parameters(): 
                    if "critic" not in pname and param.grad is not None:
                        self.ppi[tasknum][pname].add_((param.grad.detach().clone()).pow(2))

        for pname in self.ppi[tasknum].keys():
            self.ppi[tasknum][pname].mul_(1.0 / count)
        for pname, param in self.named_parameters():
            if pname in self.ppi[tasknum].keys():
                self.snapshots[tasknum][pname] = param.detach().clone()

class PPO_EWC(PPO):

    def compute_regularization_loss(self):
        ret = torch.tensor(0.0).to(self.device)
        if not hasattr(self.actor_critic, "ppi"):
            return ret
        if not hasattr(self.actor_critic, "steps"):
            self.actor_critic.steps = 0
        
        self.actor_critic.steps += 1
        num_penalties = len(self.actor_critic.ppi.keys())
        for tasknum in range(num_penalties):
            curr_penalty = torch.tensor(0.0).to(self.device)
            for pname, param in self.actor_critic.named_parameters():
                # if "actor.0" in pname: continue
                if pname in self.actor_critic.ppi[tasknum].keys():
                    param_n = self.actor_critic.snapshots[tasknum][pname]
                    ppi_n = self.actor_critic.ppi[tasknum][pname]
                    curr_penalty += ((param - param_n).pow(2) * ppi_n).sum()
            if self.actor_critic.steps % self.actor_critic.step_every == 0:
                ret += curr_penalty * self.actor_critic.multiplier
        return ret