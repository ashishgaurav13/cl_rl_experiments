import utils
utils.nowarnings()
import sys

import utils.torch, learn

import numpy as np
import gym, torch
import collections, argparse, os
from itertools import permutations, chain


def create_continual_schedule(all_envs, wts = None, rep = 1):
    if wts == None: wts = [1 for i in range(len(all_envs))]
    assert(len(all_envs) == len(wts))
    env_id_rep = list(chain(*[[env_id for i in range(wts[env_id])] for env_id in all_envs.keys()]))
    env_ids = []
    for i in range(rep): env_ids += env_id_rep[:]
    env_ids = np.random.permutation(list(env_ids))
    return env_ids

def create_eval_envs(all_envs, time_limit = 400, seed = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_envs = []
    for eid, (ob_rms_fname, env) in all_envs.items():
        [ob_rms] = torch.load(ob_rms_fname)
        eval_env = env()
        eval_env = utils.env.wrap_env(
            eval_env,
            action_normalize = True,
            time_limit = time_limit,
            deterministic = True,
            seed = seed
        )
        env_fn = lambda: eval_env
        assert(ob_rms != None)
        envs = utils.env.vectorize_env(
            [env_fn],
            state_normalize = True,
            device = device,
            train = False,
            ob_rms = ob_rms
        )
        eval_envs += [envs]

    return eval_envs


def print_state_dict(s, policy):
    if policy == None: return
    print("")
    print(s)
    state_dict = policy.state_dict()
    with torch.no_grad():
        for k, v in state_dict.items():
            s = torch.sum(v).cpu().numpy()
            mn = torch.min(torch.abs(v)).cpu().numpy()
            mx = torch.max(torch.abs(v)).cpu().numpy()
            print("%s => %s (min:%s, max:%s)" % (k, s, mn, mx))
    print("")

# verbosity = 0 => TODO
# verbosity = 1 => minimal information (MeanR, num_steps, eval_rewards)
# verbosity = 2 => all training information
def train_ppo(env_class, steps, track_eps = 25, log_interval = 1, solved_at = 90.0,
    num_processes = 8, gamma = 0.99, MaxT = 400, num_steps = 128, clip_param = 0.3,
    linear_schedule = True, policy = None, ob_rms = None, eval_envs = None,
    eval_eps = -1, hidden = -1, entropy_coef = 0, linear_schedule_mode = 0, lr = 3e-4,
    training_seed = 0, verbosity = 1):
    assert(verbosity in [1, 2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_env_steps = int(steps)
    if eval_envs != None: assert(eval_eps > 0)

    def env_fn(i):
        env = env_class()
        env = utils.env.wrap_env(
            env,
            action_normalize = True,
            time_limit = MaxT,
            deterministic = True,
            seed = i,
        )
        return lambda: env
    
    envs = utils.env.vectorize_env(
        [env_fn(i) for i in range(num_processes)],
        state_normalize = True,
        device = device,
        train = True,
    )
    if ob_rms != None: envs.ob_rms = ob_rms

    obs_space, action_space = envs.observation_space, envs.action_space
    init_obs = envs.reset()

    torch.manual_seed(training_seed)
    agent = learn.PPO(
        obs_space,
        action_space,
        init_obs,
        clip_param = clip_param,
        num_steps = num_steps,
        lr = lr,
        num_processes = num_processes,
        gamma = gamma,
        policy = policy,
        hidden = hidden,
        linear_schedule = linear_schedule,
        entropy_coef = entropy_coef,
        linear_schedule_mode = linear_schedule_mode
    )

    num_updates = agent.compute_updates_needed(num_env_steps, num_processes)
    episode_rewards = collections.deque(maxlen = track_eps)
    s = collections.deque(maxlen = track_eps)
    log_dict = {'r': episode_rewards, 'eps_done': 0, 'satisfactions': s}
    start = utils.timer()
    ret_steps = -1

    for j in range(num_updates):

        agent.pre_step(j, num_updates)
        agent.step(envs, log = log_dict)
        vloss, piloss, ent = agent.train()

        if (j+1) % log_interval == 0 and len(log_dict['r']) > 1:

            total_num_steps = (j + 1) * num_processes * num_steps
            elapsed = "Elapsed %s" % utils.timer_done(start)

            MeanR = np.mean(log_dict['r'])
            MedR = np.median(log_dict['r'])
            MinR = np.min(log_dict['r'])
            MaxR = np.max(log_dict['r'])
            if verbosity == 1:
                reward_stats = "MeanR:%.2f" % (MeanR)
                extra_stats = [reward_stats]
            elif verbosity == 2:
                reward_stats1 = "MeanR,MedR:%.2f,%.2f" % (MeanR, MedR)
                reward_stats2 = "MinR,MaxR:%.2f,%.2f" % (MinR, MaxR)
                loss_stats = "Ent:%f, VLoss:%f, PiLoss:%f" % (ent, vloss, piloss)
                extra_stats = [
                    reward_stats1,
                    reward_stats2,
                    loss_stats,
                ]
            reasons = "Reasons: %s" % (set(list(s)))
            stats = [
                "Steps:%g" % total_num_steps,
                "Eps:%d" % log_dict['eps_done'],
                elapsed,
                *extra_stats,
            ]
            print(" ".join(stats))
            print(reasons)
            if eval_envs != None:
                eval_rews = []
                for eval_env in eval_envs:
                    eval_rews += [utils.env.evaluate_ppo(agent.actor_critic, None,
                        eval_env, device, num_episodes = 1, wrap = False, silent = True)]
                print(eval_rews)
                # print("")
            sys.stdout.flush()

            if MeanR >= solved_at:
                print("Model solved! Continue")
                ret_steps = total_num_steps
                break
    
    if ret_steps == -1: print("Not solved.")
    ob_rms = utils.env.get_ob_rms(envs)
    assert(ob_rms != None)
    envs.close()
    return agent.actor_critic, ob_rms, ret_steps