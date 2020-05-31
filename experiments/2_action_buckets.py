import utils
utils.nowarnings()

import sys; sys.path.append('wm2')
import wm2.craft as craft
import wm2.tools.misc as utilities
import utils.torch, utils.env, utils, learn
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import os

all_envs = {
    0: ("models/ob_rms/osco.pt", craft.OneStoppedCarOEnv),
    1: ("models/ob_rms/osc.pt", craft.OneStoppedCarEnv),
    2: ("models/ob_rms/2sc.pt", craft.TwoStoppedCarsEnv),
    3: ("models/ob_rms/3sc.pt", craft.ThreeStoppedCarsSSO),
}
num_episodes = 10
render = False
discrete = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
intervals = 10
all_buckets = {}
bmax = 0
plt.figure(figsize = (10, 6))

for env_id, (model_name, env) in all_envs.items():
    eval_env = env(discrete = discrete)
    eval_env.debug['action_buckets'] = True
    if not os.path.isfile(model_name):
        print("Not trained, run with -train")
        exit(0)
    policy = utils.torch.load_empty_policy(learn.PolicyPPO,
        "models/gym_spaces.pt", hidden = 64)
    ob_rms = policy.load_model(model_name)
    utils.env.evaluate_ppo(policy, ob_rms, eval_env,
        device, num_episodes = num_episodes, render = render, discrete = discrete)
    print(env.__name__)
    buckets = eval_env.get_action_buckets(intervals = intervals)
    all_buckets[env.__name__] = buckets
    print(env.__name__)
    print(buckets)
    bmax = max(bmax, np.max(buckets))

for env_id, (env, bucket) in enumerate(all_buckets.items()):
    scatterx, scattery, scatters = [], [], []
    for bx in range(intervals):
        for by in range(intervals):
            if bucket[bx][by] > 0:
                s = 100. * (bucket[bx][by] / bmax)
                scatters.append(s)
                scatterx.append(bx)
                scattery.append(by)
    plt.scatter(scatterx, scattery, label = env, s = scatters)

plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
xr = map(lambda x: round(x, 2), np.arange(-2 + 2/intervals, 2, 4 / intervals))
yr = map(lambda x: round(x, 2), np.arange(-1 + 1/intervals, 1, 2 / intervals))
plt.xticks(range(intervals), xr)
plt.yticks(range(intervals), yr)
plt.subplots_adjust(right=0.35)
plt.tight_layout()
plt.show()