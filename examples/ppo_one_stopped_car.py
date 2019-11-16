import utils
utils.nowarnings()

import sys; sys.path.append('wm2')
import wm2.craft as craft
import wm2.tools.misc as utilities
import utils.torch, utils.env, learn

import numpy as np
import gym, torch
import collections, argparse, os

# Steps:89088 Eps:1172 Elapsed 0:01:26
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(agent, eval_env, model_name, num_episodes = 10, render = False):

    global device
    if not os.path.isfile(model_name):
        print("Not trained, run with -train")
        exit(0)
    ob_rms = agent.actor_critic.load_model(model_name)
    utils.env.evaluate_ppo(agent.actor_critic, ob_rms, eval_env,
        device, num_episodes = num_episodes, render = render)


def train(agent, envs, num_updates, model_name, track_eps = 25, log_interval = 1,
    solved_at = 90.0):

    episode_rewards = collections.deque(maxlen = track_eps)
    s = collections.deque(maxlen = track_eps)
    log_dict = {'r': episode_rewards, 'eps_done': 0, 'satisfactions': s}
    start = utils.timer()

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
            reward_stats1 = "MeanR,MedR:%.2f,%.2f" % (MeanR, MedR)
            reward_stats2 = "MinR,MaxR:%.2f,%.2f" % (MinR, MaxR)
            loss_stats = "Ent:%f, VLoss:%f, PiLoss:%f" % (ent, vloss, piloss)
            reasons = "Reasons: %s" % (set(list(s)))
            stats = [
                "Steps:%g" % total_num_steps,
                "Eps:%d" % log_dict['eps_done'],
                elapsed,
                reward_stats1,
                reward_stats2,
                loss_stats,
                reasons,
            ]
            print(" ".join(stats))

            if MeanR >= solved_at:
                print("Model solved!")
                ob_rms = utils.env.get_ob_rms(envs)
                assert(ob_rms != None)
                if not os.path.exists("models"): os.mkdir("models")
                agent.actor_critic.save_model(model_name, ob_rms)
                exit(0)

parser = argparse.ArgumentParser()
parser.add_argument("-train", default = False, action = "store_true")
parser.add_argument("-evaluate", default = False, action = "store_true")
parser.add_argument("-render", default = False, action = "store_true")
parser.add_argument("-file", default = "models/one_stopped_car.pt",
    help = "(default file is models/one_stopped_car.pt)")
args = parser.parse_args()
if not (args.train or args.evaluate):
    parser.print_help()
    exit(0)

num_processes = 8
gamma = 0.99
MaxT = 400 # Max number of env steps
num_env_steps = int(2e6)
num_steps = 128
log_interval = 1
# utils.log('one_stopped_car')

def env_fn(i):
    env = craft.OneStoppedCarEnv()
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

obs_space, action_space = envs.observation_space, envs.action_space
init_obs = envs.reset()

agent = learn.PPO(
    obs_space,
    action_space,
    init_obs,
    num_mini_batch = 32,
    # clip_param = 0.2,
    num_steps = num_steps,
    num_processes = num_processes,
    gamma = gamma,
)

if args.train:
    assert(not args.render)
    num_updates = agent.compute_updates_needed(num_env_steps, num_processes)
    train(agent, envs, num_updates, args.file)
elif args.evaluate:
    eval_env = craft.OneStoppedCarEnv()
    evaluate(agent, eval_env, args.file, render = args.render)