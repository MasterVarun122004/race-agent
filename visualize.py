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
FONT = pygame.font.SysFont(None, 36)
state = "MENU"
selected_option = None

# Each option maps to an action/identifier ID
options = [
    ("option_1", pygame.Rect(250, 200, 300, 50)),
    ("option_2", pygame.Rect(250, 280, 300, 50))
]

track = {}
track["outer_ellipse"] = (100, 100, 600, 400)
track["inner_ellipse"] = (180, 160, 440, 280)
track["center_x_co-ordinate"] = 400
track["center_y_co-ordinate"] = 300
track["radii_horizontal"] = 240
track["radii_vertical"] = 140

track_2 = {}
track_2["outer_ellipse"] = (100, 100, 600, 400)
track_2["inner_ellipse"] = (180, 160, 440, 280)
track_2["center_x_co-ordinate"] = 400
track_2["center_y_co-ordinate"] = 300
track_2["radii_horizontal"] = 260
track_2["radii_vertical"] = 160

tracks = {
    "option_1": track,
    "option_2": track_2
}

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if state == "MENU":
                for opt_id, rect in options:
                    if rect.collidepoint(event.pos):
                        selected_option = opt_id
                        state = "DISPLAY"
                        observation, info = env.reset()

    window.fill((0, 0, 0))

    if state == "MENU":
        for i, (opt_id, rect) in enumerate(options, 1):
            pygame.draw.rect(
                window, (50, 150, 255), rect, border_radius=8
            )
            label = FONT.render(
                f"Click Option {i}", True, (255, 255, 255)
            )
            window.blit(label, (rect.x + 70, rect.y + 12))

    elif state == "DISPLAY":
        active_track = tracks[selected_option]

        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        angle = observation[4] * 2 * np.pi

        car_x = (
            active_track["center_x_co-ordinate"]
            + active_track["radii_horizontal"] * np.cos(angle)
        )
        car_y = (
            active_track["center_y_co-ordinate"]
            + active_track["radii_vertical"] * np.sin(angle)
        )

        pygame.draw.ellipse(
            window, (100, 100, 100), active_track["outer_ellipse"]
        )
        pygame.draw.ellipse(
            window, (0, 0, 0), active_track["inner_ellipse"]
        )

        pygame.draw.rect(
            window,
            (255, 0, 0),
            (int(car_x) - 10, int(car_y) - 15, 20, 30)
        )

        label = FONT.render(
            f"Track: {selected_option}", True, (255, 255, 255)
        )
        window.blit(label, (20, 20))

    pygame.display.flip()
    clock.tick(30)
    
pygame.quit()