from gymnasium.envs.registration import register

#register(
#     id="gym_examples/GridWorld-v0",
#     entry_point="gym_examples.envs:GridWorldEnv",
#     max_episode_steps=300,
#)

register(
     id="gym_examples/Trader-v0",
     entry_point="gym_examples.envs:TraderEnv",
     max_episode_steps=1000,
)

register(
     id="gym_examples/Trader-v1",
     entry_point="gym_examples.envs:TraderEnvCnn",
     max_episode_steps=1000,
)
