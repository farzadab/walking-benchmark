from mocca_envs import register

register(
    id="EarlyStopWalker2D-v0",
    entry_point="envs.walker:EarlyStopWalker",
    max_episode_steps=1000,
)

register(
    id="EarlyStopWalker2D-v2",
    entry_point="envs.walker:EarlyStopWalker2",
    max_episode_steps=1000,
)

register(
    id="EarlyStopWalker2D-v3",
    entry_point="envs.walker:EarlyStopWalker3",
    max_episode_steps=1000,
)

register(
    id="StateEarlyStopWalker2D-v0",
    entry_point="envs.walker:StateEarlyStopWalker",
    max_episode_steps=1000,
)

register(
    id="MirrorWalker2D-v0",
    entry_point="envs.mirror:MirrorWalker2D",
    max_episode_steps=1000,
)

register(
    #
    id="MirrorPong-v0",
    entry_point="envs.mirror:MirrorPong",
    max_episode_steps=1000,
)

register(
    #
    id="MirrorHumanoid-v0",
    entry_point="envs.mirror:MirrorHumanoid",
    max_episode_steps=1000,
)

register(
    #
    id="MirrorCassieOSU-v0",
    entry_point="envs.mirror:MirrorCassieOSU",
    max_episode_steps=1000,
)

register(
    #
    id="MirrorCassieOSU2D-v0",
    entry_point="envs.mirror:MirrorCassieOSU2D",
    max_episode_steps=1000,
)
