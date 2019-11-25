RL experiments in Pytorch for catastrophic forgetting.

**RL Algorithms**:
* Deep Deterministic Policy Gradient ([ikostrikov/pytorch-ddpg-naf](https://github.com/ikostrikov/pytorch-ddpg-naf/))
* Twin Delayed DDPG ([sfujim/TD3](https://github.com/sfujim/TD3))
* PPO ([ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail))

**Requirements**: Install using `pip3 install -r requirements.txt --user`

**Models**: Saved models in `models`

**Statistics for Environments with PPO (8 processes, clip = 0.3)**:
* `NoStoppedCarEnv`: `27648 steps, 525 episodes, 0:00:22`
* `OneStoppedCarOEnv`: `21504 steps, 366 episodes, 0:00:21`
* `OneStoppedCarEnv`: `20480 steps, 473 episodes, 0:00:18`
* `TwoStoppedCarsEnv`: `100352 steps, 1942 episodes, 0:01:57`
* `ThreeStoppedCarsSSOEnv`: `130048 steps, 2437 episodes, 0:02:00`

To reproduce, run `python examples/ppo_ENV_NAME.py`. If you have an x-server error, re-run through `xvfb-run`.

**Experiments**:
* Effect of network size vs number of steps needed to solve, for various environments
* Baseline for `1SC-O, 1SC, 2SC, 3SC-SSO` in sequence.
* Baseline for `1SC-O, 1SC, 2SC, 3SC-SSO`, x 4 in a random schedule.