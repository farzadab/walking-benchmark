import torch as th
import copy
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

        ctraj.data_map["act"][:, neg_act_inds] *= -1
        swap_lr(ctraj.data_map["act"], right_act_inds, left_act_inds)

        return ctraj

    return mirror_function
