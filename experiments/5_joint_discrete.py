import utils
utils.nowarnings()

import sys; sys.path.append('wm2')
import wm2.craft as craft
import wm2.tools.misc as utilities
import utils.torch, utils.env, learn
import torch
import json
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-seed", type = int, default = 0)
args = parser.parse_args()

all_envs = {
    0: ("models/ob_rms/osco.pt", craft.OneStoppedCarOEnv),
    1: ("models/ob_rms/osc.pt", craft.OneStoppedCarEnv),
    2: ("models/ob_rms/2sc.pt", craft.TwoStoppedCarsEnv),
    3: ("models/ob_rms/3sc.pt", craft.ThreeStoppedCarsSSO),
}
eps = 0.3
entropy_coef = 0.01
lr = 3e-4
hidden = 256
eval_eps = 10
track_eps = 100
policy = None
seed = args.seed; print("Seed = %d" % seed)
eval_envs = learn.create_eval_envs(all_envs, seed = seed, discrete = True)
np.random.seed(seed)
steps = 2e5

env_classes = list(map(lambda x: x[1][1], all_envs.items()))
ob_rms = list(map(lambda x: torch.load(x[1][0])[0], all_envs.items()))
policy, _, solved_steps = learn.train_joint_ppo(env_classes, steps, policy = policy, track_eps = track_eps,
    ob_rms = ob_rms, hidden = hidden, clip_param = eps, entropy_coef = entropy_coef,
    linear_schedule = False, lr = lr, eval_envs = eval_envs, eval_eps = eval_eps,
    training_seed = seed, training_method = learn.PPO, policy_class = learn.PolicyPPO,
    verbosity = 2, discrete = True)

if solved_steps <= 0:
    print("Couldn't solve, quit.")