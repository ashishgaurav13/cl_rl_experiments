import utils
utils.nowarnings()
import sys

import utils.torch, learn

import numpy as np
import gym, torch
import collections, argparse, os
from itertools import permutations, chain

# verbosity = 0 => TODO
# verbosity = 1 => minimal information (MeanR, num_steps, eval_rewards)
# verbosity = 2 => all training information
def train_joint_ppo(env_classes, steps, track_eps = 25, log_interval = 1, solved_at = 90.0,
    num_processes = 8, gamma = 0.99, MaxT = 400, num_steps = 128, clip_param = 0.3,
    linear_schedule = True, policy = None, ob_rms = None, eval_envs = None,
    eval_eps = -1, hidden = -1, entropy_coef = 0, linear_schedule_mode = 0, lr = 3e-4,
    training_seed = 0, verbosity = 1, training_method = learn.PPO, log_extras = {},
    policy_class = learn.PolicyPPO, discrete = False):

    assert(verbosity in [1, 2])
    assert(type(env_classes) == list)
    n = len(env_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_env_steps = int(steps)
    if eval_envs != None: assert(eval_eps > 0)

    def env_fn(i, env_class):
        env = env_class(discrete = discrete)
        # env.debug['show_reasons'] = True
        env = utils.env.wrap_env(
            env,
            action_normalize = not discrete,
            time_limit = MaxT,
            deterministic = True,
            seed = i,
        )
        return lambda: env
    
    envs = [
        utils.env.vectorize_env(
            [env_fn(i, env_class) for i in range(num_processes)],
            state_normalize = True,
            device = device,
            train = True,
        ) for env_class in env_classes
    ]
    if ob_rms != None:
        assert(type(ob_rms) == list)
        for i, ob_rms_i in enumerate(ob_rms):
            envs[i].ob_rms = ob_rms_i

    obs_space, action_space = envs[0].observation_space, envs[0].action_space
    for i in range(n):
        assert(obs_space.shape == envs[i].observation_space.shape)
        assert(action_space.n == envs[i].action_space.n)

    curr_env_idx = 0
    init_obs = [envs[i].reset() for i in range(n)]

    torch.manual_seed(training_seed)
    print("training_method = %s" % training_method.__name__)
    agent = training_method(
        obs_space,
        action_space,
        init_obs[0],
        clip_param = clip_param,
        num_steps = num_steps,
        lr = lr,
        num_processes = num_processes,
        gamma = gamma,
        policy = policy,
        hidden = hidden,
        linear_schedule = linear_schedule,
        entropy_coef = entropy_coef,
        linear_schedule_mode = linear_schedule_mode,
        policy_class = policy_class
    )

    num_updates = agent.compute_updates_needed(num_env_steps, num_processes)
    episode_rewards = collections.deque(maxlen = track_eps)
    s = collections.deque(maxlen = track_eps)
    log_dict = {'r': episode_rewards, 'eps_done': 0, 'satisfactions': s, **log_extras}
    start = utils.timer()
    ret_steps = -1

    for j in range(num_updates):

        agent.pre_step(j, num_updates)
        agent.step(envs[curr_env_idx], log = log_dict)
        curr_env_idx = (curr_env_idx + 1) % n
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
                reg_loss = None
                if type(ent) == list:
                    ent, reg_loss = ent
                loss_stats = "Ent:%f, VLoss:%f, PiLoss:%f" % (ent, vloss, piloss)
                if reg_loss is not None: loss_stats += ", Reg:%f" % (reg_loss)
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
                        eval_env, device, num_episodes = eval_eps, wrap = False, silent = True)]
                    eval_rews[-1] = round(eval_rews[-1], 2)
                    eval_MeanR = np.mean(np.clip(eval_rews, -100., 100.))
                print(eval_rews)
                # print("")
            sys.stdout.flush()

            if MeanR >= solved_at:
                if eval_envs != None:
                    if eval_MeanR < solved_at:
                        continue

                print("Model solved! Continue")
                ret_steps = total_num_steps
                break
    
    if ret_steps == -1: print("Not solved.")
    ob_rms = [utils.env.get_ob_rms(envs[i]) for i in range(n)]
    for i in range(n): envs[i].close()
    return agent.actor_critic, ob_rms, ret_steps