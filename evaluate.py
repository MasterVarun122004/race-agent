
from stable_baselines3 import PPO
from environment.racing_env import RacingEnv

model = PPO.load("ppo_racing")
env = RacingEnv()

episodes = 10
successes = 0

for episode in range(episodes):
    observation, info = env.reset()
    total_reward = 0

    for step in range(1000):
        action, _ = model.predict(observation, deterministic=False)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    completed = info.get("completed") == "Lap completed"

    if completed:
        successes += 1

    print(
        f"Episode {episode + 1}: "
        f"{'Completed' if completed else 'Failed'}, "
        f"Steps={step + 1}, "
        f"Reward={total_reward:.2f}"
    )

print(f"\nSuccess rate: {successes}/{episodes} ({successes / episodes * 100:.0f}%)")

env.close()
