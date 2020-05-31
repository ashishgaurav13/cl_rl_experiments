import torch
import torch.nn as nn
import torch.optim as optim
import utils.torch
import numpy as np

class PolicyPPO(nn.Module):
    
    def __init__(self, obs_shape, action_space, hidden_size = 100):
        super(PolicyPPO, self).__init__()

        num_inputs = obs_shape[0]
        init_ = lambda m: utils.torch.init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.init_ = init_

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

    def reinit_critic(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic = nn.Sequential(
            self.init_(nn.Linear(self.num_inputs, self.hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(self.hidden_size, self.hidden_size)), nn.Tanh()).to(device)
        self.critic_linear = self.init_(nn.Linear(self.hidden_size, 1)).to(device)

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
    
    def save_model(self, path, ob_rms = None):
        print("Saving model to: %s" % path)
        torch.save([self.state_dict(), ob_rms], path)
    
    def load_model(self, path):
        print("Loading model: %s" % path)
        state_dict, ob_rms = torch.load(path)
        self.load_state_dict(state_dict)
        self.eval()
        return ob_rms