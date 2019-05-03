from mocca_envs import register

register(
    id="EarlyStopWalker2D-v0",
    entry_point="envs.walker:EarlyStopWalker",
    max_episode_steps=1000,
)
