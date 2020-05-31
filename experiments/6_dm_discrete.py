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
parser.add_argument("-const", type = float, default = 1)
parser.add_argument("-traj", type = int, default = 100)
parser.add_argument("-step_every", type = int, default = 1)
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
schedule = [0, 1, 2, 3]
steps = 2e5
trajnum = args.traj
multiplier = args.const
step_every = args.step_every

print("Schedule: %s" % schedule)
print("Multiplier = %g" % (multiplier))
for sid, env_id in enumerate(schedule):

    # if os.path.exists("model%g.pt" % (sid)):
    #     print("Already solved task %g" % env_id)
    #     policy = utils.torch.load_empty_policy(learn.PolicyPPO_EWC,
    #         "models/gym_spaces_discrete_13.pt", hidden = 256)
    #     policy_params = torch.load("model%g.pt" % (sid))
    #     policy.load_state_dict(policy_params)
    #     policy.eval()
    #     traj_mem = torch.load("buffer%g.pt" % (sid))
    #     policy.snapshot(traj_mem, multiplier = multiplier, step_every = step_every)
    #     continue

    traj_logger = utils.env.TrajectoryLogger()
    ob_rms_fname, env = all_envs[env_id]
    print("(%g) TASK %g: %s" % (sid, env_id, env.__name__))
    [ob_rms] = torch.load(ob_rms_fname)
    policy, _, solved_steps = learn.train_ppo(env, steps, policy = policy, track_eps = track_eps,
        ob_rms = ob_rms, hidden = hidden, clip_param = eps, entropy_coef = entropy_coef,
        linear_schedule = False, lr = lr, eval_envs = eval_envs, eval_eps = eval_eps,
        training_seed = seed, training_method = learn.PPO_DM, policy_class = learn.PolicyPPO_DM,
        log_extras = {"traj": traj_logger}, verbosity = 2, discrete = True, care_about = sid+1)

    policy.reinit_critic()
    
    if solved_steps <= 0:
        print("Couldn't solve task %g, quit." % sid)
        exit(0)
    else:
        policy.snapshot(traj_logger.memory, multiplier = multiplier, step_every = step_every)
        # torch.save(policy.state_dict(), "model%g.pt" % (sid))
        # torch.save(traj_logger.memory, "buffer%g.pt" % (sid))