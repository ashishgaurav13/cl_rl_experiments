from torch.utils.tensorboard import SummaryWriter

class Summary:

    def __init__(self, name, params = None):
        if params is None: params = ['R']
        assert(type(params) == list)
        self.tb = SummaryWriter(name)
        self.vals = {k: None for k in params}
        self.i = 0

    def log(self, v):
        assert(type(v) == list)
        for ki, k in enumerate(self.vals.keys()):
            self.vals[k] = v[ki]
            self.tb.add_scalar(k, v[ki], self.i)
        self.tb.flush()
        self.i += 1