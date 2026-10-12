from gymnasium.utils.env_checker import check_env
from environment.racing_env import RacingEnv

env = RacingEnv()
check_env(env)
print("Environment check passed!")


    