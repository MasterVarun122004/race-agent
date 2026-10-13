import pygame
import numpy as np
from stable_baselines3 import PPO
from environment.racing_env import RacingEnv

pygame.init()

window = pygame.display.set_mode((800,600))
pygame.display.set_caption("RL Racing Agent")
model = PPO.load("ppo_racing")
env = RacingEnv()
observation, info = env.reset()
clock = pygame.time.Clock()
running = True

while running:
    window.fill((0, 0, 0))
    observation, reward, terminated, truncated, info = env.step(
    model.predict(observation, deterministic=True)[0]
)
    angle = observation[4] * 2 * np.pi
    car_x = 400 + 240 * np.cos(angle)
    car_y = 300 + 140 * np.sin(angle)
    if terminated or truncated:
     observation, info = env.reset()
    pygame.draw.ellipse(window, (100,100,100), (100, 100, 600, 400))
    pygame.draw.ellipse(window, (0,0,0), (180, 160, 440, 280))
    pygame.draw.rect(
    window, (255, 0, 0),
    (int(car_x) - 10, int(car_y) - 15, 20, 30)
)
    pygame.display.flip()
    clock.tick(30)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()