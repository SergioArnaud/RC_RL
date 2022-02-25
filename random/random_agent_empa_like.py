import imageio
import cv2
import csv
import gym
import numpy as np
from tqdm import tqdm
import time
import uuid
import os
import random

from random_agent import AtariPreprocessing

# Initate game

n_steps = 1_000
n_seeds = 1

games = [
    "SpaceInvadersNoFrameskip-v4",
    #'BreakoutNoFrameskip-v4',
    #'FreewayNoFrameskip-v4',
    #'CarnivalNoFrameskip-v4',
    #'PongNoFrameskip-v4'
]

log_video = True
output_name = 'random_policy_empa_like'


if __name__ == "__main__":

    for game in tqdm(games):
        for n_seed in range(n_seeds):
            env = gym.make(game)
            env.reset()

            AP = AtariPreprocessing(
                env,
                terminal_on_life_loss=False,
                game_name=game.replace("NoFrameskip-v4", ""),
                tag=n_seed,
                log_video = log_video
            )
            lives = env.ale.lives()

            i = 0
            while i < n_steps:
                press_type = random.choice(["long", "short"])
                action = env.action_space.sample()

                if press_type == "long":
                    for _ in range(15):
                        observation, reward, game_over, info = AP.step(action)
                        i += 1
                        if game_over:
                            AP.reset()
                else:
                    for _ in range(4):
                        observation, reward, game_over, info = AP.step(action)
                        i += 1
                        if game_over:
                            AP.reset()
                    for _ in range(11):
                        observation, reward, game_over, info = AP.step(0)
                        i += 1
                        if game_over:
                            AP.reset()
