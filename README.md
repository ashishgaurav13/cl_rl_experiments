RL experiments in Pytorch for catastrophic forgetting.

**RL Algorithms**:
* Deep Deterministic Policy Gradient ([ikostrikov/pytorch-ddpg-naf](https://github.com/ikostrikov/pytorch-ddpg-naf/))
* Twin Delayed DDPG ([sfujim/TD3](https://github.com/sfujim/TD3))
* PPO ([ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail))

**Requirements**: Install using `pip3 install -r requirements.txt --user`<br>
**Examples**: Run an example as `python examples/EXAMPLE.py`; note that some examples need `xvfb-run` <br>
**Models**: Saved models in `models`

**Environments (8 proceses, clip = 0.3)**:
* `NoStoppedCarEnv`: `21504 steps, 443 episodes, 0:00:23`
* `OneStoppedCarEnv`: `37888 steps, 592 episodes, 0:00:42`
* `TwoStoppedCarsEnv`: `100352 steps, 1942 episodes, 0:01:57`