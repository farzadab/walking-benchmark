import torch as th
import numpy as np
import copy
import gym
import gym.envs


def auto_tune_env(env_name, kwargs):
    env_specs = gym.envs.registry.env_specs[env_name]

    name_suffix = "_".join([str(k) + "_" + str(v) for k, v in kwargs.items()])

    env_id, version = env_specs.id.split("-")
    new_id = "%s__%s-%s" % (env_id, name_suffix, version)

    if new_id not in gym.envs.registry.env_specs:
        union_kwargs = copy.deepcopy(env_specs._kwargs)
        union_kwargs.update(kwargs)

        gym.envs.registration.register(
            id=new_id,
            entry_point=env_specs._entry_point,
            tags=env_specs.tags,
            max_episode_steps=env_specs.max_episode_steps,
            reward_threshold=env_specs.reward_threshold,
            kwargs=union_kwargs,
        )

    return new_id


def get_mirror_function(
    right_obs_inds,
    left_obs_inds,
    right_act_inds,
    left_act_inds,
    neg_obs_inds=[],
    neg_act_inds=[],
):
    right_obs_inds = th.LongTensor(right_obs_inds)
    left_obs_inds = th.LongTensor(left_obs_inds)
    right_act_inds = th.LongTensor(right_act_inds)
    left_act_inds = th.LongTensor(left_act_inds)
    neg_obs_inds = th.LongTensor(neg_obs_inds)
    neg_act_inds = th.LongTensor(neg_act_inds)

    def mirror_function(traj):
        ctraj = copy.deepcopy(traj)

        def swap_lr(t, r, l):
            t[:, th.cat((r, l))] = t[:, th.cat((l, r))]

        ctraj.data_map["obs"][:, neg_obs_inds] *= -1
        swap_lr(ctraj.data_map["obs"], right_obs_inds, left_obs_inds)

        ctraj.data_map["acs"][:, neg_act_inds] *= -1
        swap_lr(ctraj.data_map["acs"], right_act_inds, left_act_inds)

        return ctraj

    return mirror_function


class MirrorEnv(gym.Wrapper):
    """
        :params mirror_indices: indices used for mirroring the environment
            example:
            mirror_indices = {
                #### observation:
                "com_obs_inds": [],          # common indices (in the observation)
                "left_obs_inds": [],         # indices of the right side (in the observation)
                "right_obs_inds": [],        # indices of the left side (in the observation)
                "neg_obs_inds": [],          # common indices that should be negated (in the observation)
                "sideneg_obs_inds": [],      # side indices that should be negated (in the observation)

                #### action:
                "com_act_inds": [],          # common indices of the action
                "neg_act_inds": [],          # common indices of the action that should be negated when mirrored
                "left_act_inds": [],         # indices of the left side in the action
                "right_act_inds": [],        # indices of the right side in the action
                "sideneg_act_inds": [],      # indices of the side that should be negated
            }
        """

    def __init__(self, env, mirror_indices):
        if "neg_act_inds" not in mirror_indices:
            mirror_indices["neg_act_inds"] = []

        super().__init__(env)

        self.mirror_indices = mirror_indices
        assert len(mirror_indices["left_obs_inds"]) == len(
            mirror_indices["right_obs_inds"]
        )
        assert len(mirror_indices["left_act_inds"]) == len(
            mirror_indices["right_act_inds"]
        )
        # *_in
        ci = len(mirror_indices["com_obs_inds"])
        ni = len(mirror_indices["neg_obs_inds"])
        si = len(mirror_indices["left_obs_inds"])
        # *_out
        co = len(mirror_indices["com_act_inds"])
        no = len(mirror_indices["neg_act_inds"])
        so = len(mirror_indices["left_act_inds"])

        # make sure the sizes match the observation space
        assert (ci + ni + 2 * si) == env.unwrapped.observation_space.shape[0]
        assert (co + no + 2 * so) == env.unwrapped.action_space.shape[0]

        env.unwrapped.mirror_sizes = [
            # obs (in)
            ci,  # c_in
            ni,  # n_in
            si,  # s_in
            # act (out)
            co,  # c_out
            no,  # n_out
            so,  # s_out
        ]
        env.unwrapped.mirror_indices = {
            "left_obs_inds": list(range(ci + ni, ci + ni + si)),
            "right_obs_inds": list(range(ci + ni + si, ci + ni + 2 * si)),
            "left_act_inds": list(range(co, co + so)),
            "right_act_inds": list(range(co + so, co + 2 * so)),
            "neg_obs_inds": list(range(ci, ci + ni)),
            "neg_act_inds": [],
        }
        self.co = co
        self.no = no
        self.so = so

    def reset(self, **kwargs):
        return self.fix_obs(self.env.reset(**kwargs))

    def step(self, act_):
        action = 0 * act_

        action[self.mirror_indices["com_act_inds"]] = act_[: self.co]
        action[self.mirror_indices["neg_act_inds"]] = act_[self.co : self.co + self.no]
        action[self.mirror_indices["left_act_inds"]] = act_[-2 * self.so : -self.so]
        action[self.mirror_indices["right_act_inds"]] = act_[-self.so :]

        action[self.mirror_indices["sideneg_act_inds"]] *= -1
        obs, reward, done, info = self.env.step(action)
        return self.fix_obs(obs), reward, done, info

    def fix_obs(self, obs):
        obs[self.mirror_indices["sideneg_obs_inds"]] *= -1
        return np.concatenate(
            [
                obs[self.mirror_indices["com_obs_inds"]],
                obs[self.mirror_indices["neg_obs_inds"]],
                obs[self.mirror_indices["left_obs_inds"]],
                obs[self.mirror_indices["right_obs_inds"]],
            ]
        )
