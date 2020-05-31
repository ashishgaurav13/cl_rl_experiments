RL experiments in Pytorch for catastrophic forgetting.

**RL Algorithms**:
* PPO ([ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail))

**Requirements**: Install using `pip3 install -r requirements.txt --user`

**Statistics for Discrete Environments with PPO (8 processes, clip = 0.3)**:
* `OneStoppedCar`: `20480 steps, 473 episodes, 0:00:18`
    * stopped car on same lane
* `OneStoppedCarO`: `21504 steps, 366 episodes, 0:00:21`
* `TwoStoppedCars`: `100352 steps, 1942 episodes, 0:01:57`
* `ThreeStoppedCarsSSO`: `130048 steps, 2437 episodes, 0:02:00`

To reproduce, run `python examples/ppo_ENV_NAME.py -discrete`. If you have an x-server error, re-run through `xvfb-run`.

**Experiments**:
* Effect of network size vs number of steps needed to solve, for various environments
* Action bucketing: In continuous environments, place the actions taken into discrete buckets; then frequent actions can be used to design the discrete action space for discrete environments
* Baseline for `1SC-O, 1SC, 2SC, 3SC-SSO` in sequence.
* Elastic Weight Consolidation (EWC) for `1SC-O, 1SC, 2SC, 3SC-SSO` in sequence.
    * Step-every parameter chooses delta.
* Joint training `1SC-O, 1SC, 2SC, 3SC-SSO`.