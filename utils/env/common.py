from inspect import isfunction
import numpy as np
import sys

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
    losses = np.mean(losses, axis = 0)
    losses = np.round(losses, 6)

    if debug:
        extra_str = ""
        if history is not None:
            history += [R]
            extra_str = ", R%d:%.2f" % (smooth, np.mean(history[-smooth:]))
        print("Ep:%d, T:%d, R:%.2f%s, Loss:%s" % (episode_num, T, R, extra_str, losses))
    
    if tb != None:
        tb.log([R])

    return T, R


