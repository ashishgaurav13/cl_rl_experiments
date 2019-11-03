import numpy as np
import torch
from math import sqrt

class OUNoise:

    def __init__(self, mu, theta=0.15, sigma=0.2, dt = 1e-2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def __call__(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(*self.mu.shape)
        self.state = x + dx
        return self.state

class AdaptiveParamNoiseSpec(object):

    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient
        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            self.current_stddev /= self.adaptation_coefficient
        else:
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats