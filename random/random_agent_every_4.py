import imageio
import cv2
import csv
import gym
import numpy as np
from tqdm import tqdm
import time
import uuid
import os

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
output_name = 'random_policy'


if __name__ == "__main__":
    for game in tqdm(games):
        for n_seed in range(n_seeds):
            env = gym.make(game)
            env.reset()
            AP = AtariPreprocessing(
                env,
                terminal_on_life_loss=True,
                game_name=game.replace("NoFrameskip-v4", ""),
                tag=n_seed,
                log_video = log_video,
                output_name = output_name
            )
            action = env.action_space.sample()
            lives = env.ale.lives()
            observation, reward, game_over, info = AP.step(action)
            for i in range(n_steps):
                action = env.action_space.sample()
                observation, reward, game_over, info = AP.step(action)
                if game_over:
                    AP.reset()


"""
n_seeds = 20
scores = {}
for game in tqdm(games):
    scores[game] = []
    for n_seed in range(n_seeds):
        env = gym.make(game)
        env.reset()
        action = env.action_space.sample()
        lives = env.ale.lives()
        observation, reward, game_over, info = env.step(action)
        is_terminal = game_over 
        ac_rew = 0
        steps = 0
        while not is_terminal:
            action = env.action_space.sample()
            observation, reward, game_over, info = env.step(action)
            ac_rew += reward
            new_lives = info['ale.lives']
            is_terminal = game_over or new_lives < lives
            lives = new_lives
            steps += 1

        scores[game].append(ac_rew)

# print mean and std
for game in games:
    print('{} mean: {} std: {}'.format(game, 
                                       np.mean(scores[game]), 
                                       np.std(scores[game])
                                       )
                                       )
"""
