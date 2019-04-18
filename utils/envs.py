import copy
import gym.envs

def auto_tune_env(env_name, kwargs):
    env_specs = gym.envs.registry.env_specs[env_name]

    name_suffix = '_'.join([str(k) + '_' + str(v) for k, v in kwargs.items()])

    env_id, version = env_specs.id.split('-')
    new_id = '%s__%s-%s' % (env_id, name_suffix, version)

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