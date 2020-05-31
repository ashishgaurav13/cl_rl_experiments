
# Log successful trajectories so that we can compute per-parameter
# importance for various parameters over these states.
class TrajectoryLogger:

    def __init__(self):
        self.memory = []

    def see(self, obs, reward, done, infos):
        if not hasattr(self, "num_processes"):
            self.num_processes = obs.shape[0]
            self.trajectories = {i: [] for i in range(self.num_processes)}
        assert(self.num_processes == obs.shape[0])
        for np in range(self.num_processes):
            self.trajectories[np].append(obs[np])
            if done[np] == True:
                if "episode" in infos[np]:
                    if infos[np]["mode"] == "success":
                        self.memory.append(self.trajectories[np])
                        # print("Found a successful trajectory with R = %.2f" %
                        #     infos[np]["episode"]["r"])
                self.trajectories[np] = []