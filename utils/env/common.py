from inspect import isfunction
import numpy as np
import sys
from .gym_wrappers import *
from baselines import bench
import utils.torch
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def episode(env, policy, process_experience, train,
    episode_num = None, render = False, debug = True, tb = None,
    history = None, smooth = 50, n_updates = 1):
    assert(isfunction(policy))
    assert(isfunction(process_experience))
    assert(isfunction(train))
    if debug: assert(episode_num != None)
    obs = env.reset()
    done = False
    T = 0
    R = 0

    losses = []
    while not done:
        action = policy(obs)
        next_obs, reward, done, info = env.step(action)
        process_experience(obs, action, reward, next_obs, done)
        for _ in range(n_updates): losses += [train()]
        T += 1
        R += reward
        if render: env.render()
    try:
        losses = np.mean(losses, axis = 0)
        losses = np.round(losses, 6)
    except:
        losses = -np.inf

    if debug:
        extra_str = ""
        if history is not None:
            history += [R]
            extra_str = ", R%d:%.2f" % (smooth, np.mean(history[-smooth:]))
        if losses != -np.inf:
            print("Ep:%d, T:%d, R:%.2f%s, Loss:%s" % (episode_num, T, R, extra_str, losses))
        else:
            print("Ep:%d, T:%d, R:%.2f%s" % (episode_num, T, R, extra_str))

    if tb != None:
        tb.log([R])

    return T, R

def n_episodes(env, policy, n_episodes = 10,
    render = False, debug = True):

    assert(env != None)
    assert(isfunction(policy))
    print("Evaluating for %d episodes." % n_episodes)

    train = lambda: None
    process_experience = lambda obs, action, reward, next_obs, done: None

    for ep in range(n_episodes):

        episode(env, policy, process_experience, train,
            episode_num = ep+1, render = render, debug = True)

def get_ob_rms(envs):
    if isinstance(envs, VecNormalize):
        return envs.ob_rms
    elif hasattr(envs, 'venv'):
        return get_ob_rms(envs.venv)
    return None

def evaluate_ppo(network, ob_rms, env, device, num_episodes = 10,
    time_limit = 400, render = False):

    orig_env = env
    env = wrap_env(
        env,
        action_normalize = True,
        time_limit = time_limit,
        deterministic = True,
    )
    env_fn = lambda: env
    envs = vectorize_env(
        [env_fn],
        state_normalize = True,
        device = device,
        train = False,
        ob_rms = ob_rms
    )
    
    eval_episode_rewards = []
    orig_env.seed(len(eval_episode_rewards)+1000)
    obs = envs.reset()
    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            _, action, _ = network.act(
                obs,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = envs.step(action)
        if render: orig_env.render()

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                print("R:%.2f, T:%d" % (info['episode']['r'], info['episode']['l']))
                orig_env.seed(len(eval_episode_rewards)+1000)
                obs = envs.reset() # Only works for one env (TODO)

    # envs.close()

    print("Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))


# Wrap single env
def wrap_env(
    env,
    action_normalize = True,
    time_limit = None,
    deterministic = True,
    seed = 0):

    if action_normalize:
        env = NormalizedActions(env)
    if time_limit != None:
        env = TimeLimit(env, time_limit)
    if deterministic:
        env = utils.torch.make_deterministic(env)
    env.seed(seed)
    env = bench.Monitor(env, filename = None, allow_early_resets = True)
    return env

# Vectorize envs
def vectorize_env(
    envs,
    state_normalize = True,
    device = None,
    train = True,
    gamma = 0.99,
    ob_rms = None):

    assert(type(envs) == list)
    for env_fn in envs: assert(isfunction(env_fn))
    
    assert(len(envs) == 1)
    envs = DummyVecEnv(envs)

    if state_normalize:
        if train:
            envs = VecNormalize(envs, gamma = gamma)
        else:
            envs = VecNormalize(envs, ret = False)
            envs.eval()
            assert(ob_rms != None)
            envs.ob_rms = ob_rms
    
    if device != None:
        envs = VecPyTorch(envs, device)
    
    return envs