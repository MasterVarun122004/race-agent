from stable_baselines3 import PPO
from environment.racing_env import RacingEnv

env = RacingEnv()

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1
)

model.learn(total_timesteps=50000)

model.save("ppo_racing")