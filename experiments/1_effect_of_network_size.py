import utils
utils.nowarnings()

import sys; sys.path.append('wm2')
import wm2.craft as craft
import wm2.tools.misc as utilities
import utils.torch, utils.env, utils, learn
import torch
import json
import numpy as np

all_envs = {
    0: ("models/ob_rms/osco.pt", craft.OneStoppedCarOEnv),
    1: ("models/ob_rms/osc.pt", craft.OneStoppedCarEnv),
    2: ("models/ob_rms/2sc.pt", craft.TwoStoppedCarsEnv),
    3: ("models/ob_rms/3sc.pt", craft.ThreeStoppedCarsSSO),
}

steps = 1.5e5 # 1e6
network_sizes = [32, 50, 64, 100, 128, 150, 200, 250, 256, 300]

steps_to_solve = utils.json_load(f = "graphs/1_effect_of_network_size.txt", show = True)

for env_id in all_envs.keys():
    
    if str(env_id) not in steps_to_solve:
        steps_to_solve[str(env_id)] = {}
    
    for hidden in network_sizes:
        
        if str(hidden) in steps_to_solve[str(env_id)]:
            continue
        
        ob_rms_fname, env = all_envs[env_id]
        [ob_rms] = torch.load(ob_rms_fname)
        policy, _, total_num_steps = \
            learn.train_ppo(env, steps, ob_rms = ob_rms, hidden = hidden,
                linear_schedule = False, clip_param = 0.3)
        steps_to_solve[str(env_id)][str(hidden)] = total_num_steps

print(steps_to_solve)
# utils.json_dump(steps_to_solve, f = "effect_of_network_size.txt", show = True)