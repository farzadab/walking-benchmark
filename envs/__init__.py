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
    id="StateEarlyStopWalker2D-v0",
    entry_point="envs.walker:StateEarlyStopWalker",
    max_episode_steps=1000,
)
