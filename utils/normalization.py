import torch as th
import gym


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.stats = Stats(env.observation_space.shape, max_obs=4000)

    def replace_wrapped_env(self, env):
        self.env = env

    def observation(self, obs):
        self.stats.observe(obs)
        return self.stats.normalize(obs).numpy()

    def load_state_dict(self, state_dict):
        self.stats.mean[:] = state_dict["stats__mean"]
        self.stats.std[:] = state_dict["stats__std"]
        self.disable_update()

    def disable_update(self):
        self.stats.disable_update()

    def state_dict(self):
        return {"stats__mean": self.stats.mean, "stats__std": self.stats.std}

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


# Taken from https://gitlab.com/zxieaa/cassie.git
class Stats(th.nn.Module):
    def __init__(
        self, input_size, shift_mean=True, scale=True, clip=True, max_obs=float("inf")
    ):
        """
        @brief used for normalizing a quantity (1D)
        @param input_size: the input_sizeension of the quantity
        """
        # not really using the shared memory for now, but might be useful later
        self.shift_mean = shift_mean
        self.scale = scale
        self.clip = clip
        self.input_size = input_size
        self.max_obs = max_obs
        self.reset()

    def disable_update(self):
        self.max_obs = -1

    def observe(self, obs):
        """
        @brief update observation mean & stdev
        # @param obs: the observation. assuming NxS where N is the batch-size and S is the input-size
        @param obs: the observation (1D)
        """
        if self.n > self.max_obs:
            return
        obs = th.FloatTensor(obs)
        self.n += 1
        self.sum[:] = self.sum + obs
        self.sum_sqr += obs.pow(2)
        self.mean[:] = self.sum / self.n
        self.std[:] = (
            (self.sum_sqr / self.n - self.mean.pow(2)).clamp(1e-4, 1e10).sqrt()
        )

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
        # obs_std = th.sqrt(self.var).unsqueeze(0).expand_as(inputs)
        return normalized

    def reset(self):
        self.n = th.zeros(1).share_memory_()
        self.mean = th.zeros(self.input_size, dtype=th.float).share_memory_()
        self.std = th.ones(self.input_size, dtype=th.float).share_memory_()
        self.sum = th.zeros(self.input_size, dtype=th.float).share_memory_()
        self.sum_sqr = th.zeros(self.input_size, dtype=th.float).share_memory_()

    def get_normalization_params(self):
        return self.mean, self.std
