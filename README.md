RL experiments in Pytorch for catastrophic forgetting.

**RL Algorithms**:
* Deep Deterministic Policy Gradient ([ikostrikov/pytorch-ddpg-naf](https://github.com/ikostrikov/pytorch-ddpg-naf/))
* Twin Delayed DDPG ([sfujim/TD3](https://github.com/sfujim/TD3))
* PPO ([ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail))

**Requirements**: Install using `pip3 install -r requirements.txt --user`<br>
**Examples**: Run an example as `python examples/EXAMPLE.py`; note that some examples need `xvfb-run` <br>
**Models**: Saved models in `models`

**Environments (8 processes, clip = 0.3)**:
* `NoStoppedCarEnv`: `27648 steps, 525 episodes, 0:00:22`
* `OneStoppedCarOEnv`: `21504 steps, 366 episodes, 0:00:21`
* `OneStoppedCarEnv`: `20480 steps, 473 episodes, 0:00:18`
* `TwoStoppedCarsEnv`: `100352 steps, 1942 episodes, 0:01:57`
* `ThreeStoppedCarsSSOEnv`: `130048 steps, 2437 episodes, 0:02:00`