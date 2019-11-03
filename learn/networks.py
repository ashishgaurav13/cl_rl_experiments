import torch
import torch.nn as nn
import torch.optim as optim
import utils.torch
import numpy as np

class Actor(nn.Module):

    def __init__(self, lr, num_inputs, action_space, h = [400, 300]):

        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, h[0])
        self.linear2 = nn.Linear(h[0], h[1])
        self.mu = nn.Linear(h[1], num_outputs)
        torch.nn.init.uniform_(self.mu.weight.data, -0.003, 0.003)
        torch.nn.init.uniform_(self.mu.bias.data, -0.003, 0.003)

        self.relu, self.tanh = nn.ReLU(), nn.Tanh()
        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.mu(x)
        x = self.tanh(x)
        return x

class Critic(nn.Module):

    def __init__(self, lr, num_inputs, action_space, h = [400, 300]):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, h[0])
        self.linear2 = nn.Linear(h[0]+num_outputs, h[1])
        self.V = nn.Linear(h[1], 1)
        torch.nn.init.uniform_(self.V.weight.data, -0.0003, 0.0003)
        torch.nn.init.uniform_(self.V.bias.data, -0.0003, 0.0003)

        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, inputs, actions):
        x = inputs
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(torch.cat((x, actions), 1))
        x = self.relu(x)
        V = self.V(x)
        return V

class CriticTD3(nn.Module):

    def __init__(self, lr, num_inputs, action_space, h = [256, 256]):
        super(CriticTD3, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs+num_outputs, h[0])
        self.linear2 = nn.Linear(h[0], h[1])
        self.V = nn.Linear(h[1], 1)
        torch.nn.init.uniform_(self.V.weight.data, -0.0003, 0.0003)
        torch.nn.init.uniform_(self.V.bias.data, -0.0003, 0.0003)

        self.linear3 = nn.Linear(num_inputs+num_outputs, h[0])
        self.linear4 = nn.Linear(h[0], h[1])
        self.W = nn.Linear(h[1], 1)
        torch.nn.init.uniform_(self.W.weight.data, -0.0003, 0.0003)
        torch.nn.init.uniform_(self.W.bias.data, -0.0003, 0.0003)

        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, inputs, actions, both = True):
        x = torch.cat((inputs, actions), 1)
        if both:
            v1, v2 = self.linear1(x), self.linear3(x)
            v1, v2 = self.relu(v1), self.relu(v2)
            v1, v2 = self.linear2(v1), self.linear4(v2)
            v1, v2 = self.relu(v1), self.relu(v2)
            v1, v2 = self.V(v1), self.W(v2)
            return v1, v2
        else:
            v = self.linear1(x)
            v = self.relu(v)
            v = self.linear2(v)
            v = self.relu(v)
            v = self.V(v)
            return v

class PolicyPPO(nn.Module):
    
    def __init__(self, obs_shape, action_space, hidden_size = 64):
        super(PolicyPPO, self).__init__()

        num_inputs = obs_shape[0]
        init_ = lambda m: utils.torch.init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = utils.torch.Categorical(hidden_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = utils.torch.DiagGaussian(hidden_size, num_outputs)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        x = inputs
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        return self.critic_linear(hidden_critic), hidden_actor


    def act(self, inputs, deterministic=False):

        value, actor_features = self.forward(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs):
        x = inputs
        hidden_critic = self.critic(x)
        value = self.critic_linear(hidden_critic)
        return value

    def evaluate_actions(self, inputs, action):

        value, actor_features = self.forward(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy