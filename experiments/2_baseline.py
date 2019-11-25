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
    0: ("models/ob_rms/one_stopped_car_o.pt", craft.OneStoppedCarOEnv),
    1: ("models/ob_rms/one_stopped_car.pt", craft.OneStoppedCarEnv),
    2: ("models/ob_rms/two_stopped_cars.pt", craft.TwoStoppedCarsEnv),
    3: ("models/ob_rms/three_stopped_cars.pt", craft.ThreeStoppedCarsSSO),
}
eps = 0.3
entropy_coef = 0.01
lrs = [5e-4, 3e-4, 5e-4, 3e-4]
policy = None
seed = args.seed; print("Seed = %d" % seed)
eval_envs = learn.create_eval_envs(all_envs, seed = seed)
np.random.seed(seed)
schedule = [0, 1, 2, 3]
steps = 1.5e5

print("Schedule: %s" % schedule)
for sid, env_id in enumerate(schedule):


    ob_rms_fname, env = all_envs[env_id]
    print("(%g) TASK %g: %s" % (sid, env_id, env.__name__))
    [ob_rms] = torch.load(ob_rms_fname)
    policy, _, solved_steps = learn.train_ppo(env, steps, policy = policy, track_eps = 50,
        ob_rms = ob_rms, hidden = 256, clip_param = eps, entropy_coef = entropy_coef,
        linear_schedule = False, lr = lrs[env_id], eval_envs = eval_envs, eval_eps = 1,
        training_seed = seed)

    policy.reinit_critic()
    if solved_steps > 0:
        pass
    else:
        print("Couldn't solve task %g, quit." % sid)
        continue