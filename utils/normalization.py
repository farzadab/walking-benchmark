import torch as th

from baselines.common.vec_env import VecEnvWrapper


class NormalizedEnv(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.state_stats = Stats(input_size=self.observation_space.shape, max_obs=4000)
        self.reward_stats = Stats(input_size=1, shift_mean=False, clip=False, max_obs=4000)
    
    def reset(self):
        obs = self.venv.reset()
        self.state_stats.observe(th.FloatTensor(obs))
        return self.state_stats.normalize(obs)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.state_stats.observe(th.FloatTensor(obs))
        self.reward_stats.observe(th.FloatTensor(rews))
        return self.state_stats.normalize(obs), self.reward_stats.normalize(rews), dones, infos
    
    def get_normalization_params(self):
        return (
            self.state_stats.get_normalization_params(),
            self.reward_stats.get_normalization_params(),
        )


# Taken from https://gitlab.com/zxieaa/cassie.git
class Stats():
    def __init__(self, input_size, shift_mean=True, scale=True, clip=True, max_obs=float('inf')):
        '''
        @brief used for normalizing a quantity (1D)
        @param input_size: the input_sizeension of the quantity
        '''
        # not really using the shared memory for now, but might be useful later
        self.shift_mean = shift_mean
        self.scale = scale
        self.clip = clip
        self.input_size = input_size
        self.max_obs = max_obs
        self.reset()

    def observe(self, obs):
        '''
        @brief update observation mean & stdev
        @param obs: the observation. assuming NxS where N is the batch-size and S is the input-size
        '''
        if self.n > self.max_obs:
            return
        obs = th.FloatTensor(obs)
        self.n += obs.shape[0]
        self.sum = self.sum + obs.sum(0)
        self.sum_sqr += obs.pow(2).sum(0)
        self.mean = self.sum / self.n
        self.std = (self.sum_sqr / self.n - self.mean.pow(2)).clamp(1e-4,1e10).sqrt()
        self.mean = self.mean.float()
        self.std = self.std.float()

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = self.std
        inputs = th.FloatTensor(inputs)
        if len(inputs.shape) > 1:
            obs_mean = obs_mean.unsqueeze(0).expand_as(inputs)
            obs_std = obs_std.unsqueeze(0).expand_as(inputs)
        normalized = inputs
        if self.shift_mean:
            normalized -= obs_mean
        if self.scale:
            normalized /= obs_std
        if self.clip:
            normalized = th.clamp(normalized, -10.0, 10.0)
        # normalized = (inputs - obs_mean) / obs_std
        #obs_std = th.sqrt(self.var).unsqueeze(0).expand_as(inputs)
        return normalized

    def reset(self):
        self.n = th.zeros(1).share_memory_()
        self.mean = th.zeros(self.input_size).share_memory_()
        self.std = th.ones(self.input_size).share_memory_()
        self.sum = th.zeros(self.input_size).share_memory_()
        self.sum_sqr = th.zeros(self.input_size).share_memory_()
    
    def get_normalization_params(self):
        return self.mean, self.std