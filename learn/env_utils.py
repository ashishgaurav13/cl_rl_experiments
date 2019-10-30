from inspect import isfunction

def episode(env, policy, process_experience, train,
    episode_num = None, render = False, debug = True):
    assert(isfunction(policy))
    assert(isfunction(process_experience))
    assert(isfunction(train))
    if debug: assert(episode_num != None)
    obs = env.reset()
    done = False
    T = 0
    R = 0

    while not done:
        action = policy(obs)
        next_obs, reward, done, info = env.step(action)
        process_experience(obs, action, reward, next_obs, done)
        train()
        T += 1
        R += reward
        if render: env.render()
    
    if debug: print("Ep:%d, T:%d, R:%.2f" % (episode_num, T, R))
    return T, R