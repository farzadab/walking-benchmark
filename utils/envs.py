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
                "left_act_inds": [],         # indices of the left side in the action
                "right_act_inds": [],        # indices of the right side in the action
                "sideneg_act_inds": [],      # indices of the side that should be negated
            }
        """

    def __init__(self, env, mirror_indices):
        super().__init__(env)
        env.unwrapped.mirror_indices = mirror_indices
        assert len(mirror_indices["left_obs_inds"]) == len(
            mirror_indices["right_obs_inds"]
        )
        assert len(mirror_indices["left_act_inds"]) == len(
            mirror_indices["right_act_inds"]
        )
        env.unwrapped.mirror_sizes = [
            len(mirror_indices["com_obs_inds"]),  # c_in
            len(mirror_indices["neg_obs_inds"]),  # n_in
            len(mirror_indices["left_obs_inds"]),  # s_in
            #
            len(mirror_indices["com_act_inds"]),  # c_out
            0,  # n_out
            len(mirror_indices["left_act_inds"]),  # s_out
        ]

    def reset(self, **kwargs):
        return self.fix_obs(self.env.reset(**kwargs))

    def step(self, action):
        action[self.unwrapped.mirror_indices["sideneg_act_inds"]] *= -1
        obs, reward, done, info = self.env.step(action)
        return self.fix_obs(obs), reward, done, info

    def fix_obs(self, obs):
        obs[self.unwrapped.mirror_indices["sideneg_obs_inds"]] *= -1
        return np.concatenate(
            [
                obs[self.unwrapped.mirror_indices["com_obs_inds"]],
                obs[self.unwrapped.mirror_indices["neg_obs_inds"]],
                obs[self.unwrapped.mirror_indices["left_obs_inds"]],
                obs[self.unwrapped.mirror_indices["right_obs_inds"]],
            ]
        )
