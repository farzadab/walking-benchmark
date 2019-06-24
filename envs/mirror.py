from roboschool.gym_mujoco_walkers import RoboschoolWalker2d
from utils.envs import MirrorEnv
import gym


def MirrorWalker2D():
    return MirrorEnv(
        env=gym.make("roboschool:RoboschoolWalker2d-v1"),
        mirror_indices={
            #### observation:
            "com_obs_inds": [0, 2, 3, 5, 7],
            "neg_obs_inds": [1, 4, 6],
            "left_obs_inds": list(range(8, 14)) + [20],
            "right_obs_inds": list(range(14, 20)) + [21],
            "sideneg_obs_inds": [],
            #### action:
            "com_act_inds": [],
            "left_act_inds": list(range(0, 3)),
            "right_act_inds": list(range(3, 6)),
            "sideneg_act_inds": [],
        },
    )
