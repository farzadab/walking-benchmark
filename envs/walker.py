from roboschool.gym_mujoco_walkers import RoboschoolWalker2d
import torch as th
import numpy as np
import os
import gym


class EarlyStopWalker(RoboschoolWalker2d):
    vf_path = os.path.join(os.path.dirname(__file__), "data", "vf_max.pt")
    state_norm_path = os.path.join(
        os.path.dirname(__file__), "data", "env__state_max.pt"
    )
    threshold = 65

    def __init__(self, *args, **kwargs):
        super(EarlyStopWalker, self).__init__(*args, **kwargs)
        self.vf = th.load(self.vf_path)
        norms_dict = th.load(self.state_norm_path)
        self.obs_mean = norms_dict["stats__mean"]
        self.obs_std = norms_dict["stats__std"]

    def state_value(self, obs):
        return self.vf((th.FloatTensor(obs) - self.obs_mean) / self.obs_std)[0].item()

    def step(self, action):
        obs, rew, done, info = super(EarlyStopWalker, self).step(action)
        state_value = self.state_value(obs)
        info["state_value"] = state_value
        info["done"] = done
        if state_value < self.threshold:
            done = True
        return obs, rew, done, info

    def render(self, *args, **kwargs):
        from OpenGL import GLU  # hacky fix

        super(EarlyStopWalker, self).render(*args, **kwargs)


class EarlyStopWalker2(EarlyStopWalker):
    vf_path = os.path.join(os.path.dirname(__file__), "data", "0.1_vf_max.pt")
    state_norm_path = os.path.join(
        os.path.dirname(__file__), "data", "0.1_env__state_max.pt"
    )
    threshold = 65


class EarlyStopWalker3(EarlyStopWalker):
    vf_path = os.path.join(os.path.dirname(__file__), "data", "vf_progress.pt")
    state_norm_path = os.path.join(
        os.path.dirname(__file__), "data", "env__state_progress.pt"
    )
    threshold = 10  # TODO: tune


class StateEarlyStopWalker(RoboschoolWalker2d):
    states_path = os.path.join(os.path.dirname(__file__), "data", "walker_states.csv")
    threshold = 40

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(self.states_path, "rb") as dfile:
            self.gstates = np.loadtxt(dfile, delimiter=",")
        self.weights = np.ones(self.gstates.shape[1])
        # self.weights[0] = 10

    def distance(self, obs):
        errors = (self.gstates - obs) ** 2 * self.weights
        return errors.sum(axis=1).min()

    def step(self, action):
        obs, rew, done, info = super().step(action)
        info["done"] = done
        # done = False
        info["distance"] = self.distance(obs)
        if info["distance"] > self.threshold:
            done = True
        return obs, rew, done, info

    def render(self, *args, **kwargs):
        from OpenGL import GLU  # hacky fix

        super().render(*args, **kwargs)


def run():
    env = gym.make("EarlyStopWalker2D-v3")
    obs = env.reset()

    while True:
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())
        if done:
            bstate = env.unwrapped.state_value(obs)
            obs = env.reset()
            print(info["done"], bstate, '  new ->', env.unwrapped.state_value(obs))


if __name__ == "__main__":
    run()
