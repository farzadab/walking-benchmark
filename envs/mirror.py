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


def MirrorHumanoid():
    return MirrorEnv(
        env=gym.make("roboschool:RoboschoolHumanoid-v1"),
        mirror_indices={
            #### observation:
            "com_obs_inds": [
                0,  # z
                2,  # cos(yaw)
                3,  # vx
                5,  # vz
                7,  # pitch
                # common joints
                10,
                11,
            ],
            "neg_obs_inds": [
                1,  # sin(yaw)
                4,  # vy
                6,  # roll
                # neg joints
                8,
                9,
                12,
                13,
            ],
            "left_obs_inds": list(range(22, 30)) + list(range(36, 42)) + [43],
            "right_obs_inds": list(range(14, 22)) + list(range(30, 36)) + [42],
            "sideneg_obs_inds": list(range(30, 34)),
            #### action:
            "com_act_inds": [0, 1, 2],
            "left_act_inds": [7, 8, 9, 10, 14, 15, 16],
            "right_act_inds": [3, 4, 5, 6, 11, 12, 13],
            "sideneg_act_inds": [0, 2, 11, 12],
        },
    )


def MirrorPong():
    return MirrorEnv(
        env=gym.make("roboschool:RoboschoolPong-v1"),
        mirror_indices={
            #### observation:
            "com_obs_inds": [0, 1, 4, 5, 8, 9, 12],
            "neg_obs_inds": [2, 3, 6, 7, 10, 11],
            "left_obs_inds": [],
            "right_obs_inds": [],
            "sideneg_obs_inds": [],
            #### action:
            "com_act_inds": [0, 1],
            "left_act_inds": [],
            "right_act_inds": [],
            "sideneg_act_inds": [1],
        },
    )
