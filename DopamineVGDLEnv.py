from gym import spaces
from collections import defaultdict
import numpy as np
from scipy import misc
import imageio
import sys
from VGDLEnv import VGDLEnv
import csv
import cloudpickle
import cv2
import time
import os
from scipy import misc
import uuid



class DopamineVGDLEnv(object):
    def __init__(self, game_name, agent_name, parameter_set="", tag="", uuid = None):

        # CONFIGS
        self.game_name = game_name
        self.game_name_short = game_name[5:]
        self.level_switch = "sequential"
        self.trial_num = 1003
        self.criteria = "1/1"
        self.timeout = 2000
        games_folder = "../all_games"

        date = time.strftime("%Y.%m.%d")

        if uuid is None:
            self.experiment_uuid = uuid.uuid1()
        else:
            self.experiment_uuid = uuid

        if agent_name == "EfficientZero":
            experiment_id = "{}_{}_{}_{}".format(
                time.strftime("%Y.%m.%d_%H.%M.%S_%f"), game_name, parameter_set, tag
            )

            self.experiment_outpath = "../experiments/{}/{}/{}/{}/{}".format(
                agent_name, game_name, date, self.experiment_uuid, experiment_id
            )
        else:
            experiment_id = "{}_{}_{}_{}_{}".format(
                time.strftime("%Y.%m.%d_%H.%M.%S_%f"), game_name, parameter_set, tag, self.experiment_uuid 
            )

            self.experiment_outpath = "../experiments/{}/{}/{}/{}".format(
                agent_name, game_name, date, experiment_id
            )

        os.makedirs(self.experiment_outpath, exist_ok=True)
        os.makedirs(self.experiment_outpath + "/screens", exist_ok=True)
        os.makedirs(self.experiment_outpath + "/object_positions", exist_ok=True)
        os.makedirs(self.experiment_outpath + "/avatar_positions", exist_ok=True)

        # FOR RECORDING
        self.record_flag = 1  # record_flag
        # pdb.set_trace()

        self.avatar_file_path = "{}/avatar_positions/{}_{}_avatar".format(
            self.experiment_outpath, self.experiment_uuid, self.game_name_short
        )
        self.objects_file_path = "{}/object_positions/{}_{}_objects".format(
            self.experiment_outpath, self.experiment_uuid, self.game_name_short
        )

        self.Env = VGDLEnv(self.game_name_short, games_folder)
        self.Env.set_level(0)
        self.action_space = spaces.Discrete(len(self.Env.actions))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(84, 84, 3))

        self.game_over = 0
        self.screen_history = []
        self.steps = 0
        self.episode_steps = 0
        self.episode = 0
        self.episode_reward = 0
        self.event_dict = defaultdict(lambda: 0)
        self.recent_history = [0] * int(self.criteria.split("/")[1])

        self.avatar_position_data = {
            "game_info": (
                self.Env.current_env._game.width,
                self.Env.current_env._game.height,
            ),
            "data":[]
        }

        self.objects_position_data = {
            "game_info": (
                self.Env.current_env._game.width,
                self.Env.current_env._game.height,
            ),
            "data": []
        }
        

        if self.record_flag:
            with open(
                "{}/{}_{}_reward_history_{}.csv".format(
                    self.experiment_outpath, 
                    self.experiment_uuid,
                    self.game_name_short, 
                    self.level_switch
                ),
                "w",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["level", "steps", "ep_reward", "win", "game_name", "criteria"]
                )

            with open(
                "{}/{}_{}_object_interaction_history_{}.csv".format(
                    self.experiment_outpath,
                    self.experiment_uuid,
                    self.game_name_short, 
                    self.level_switch
                ),
                "a",
            ) as file:
                interactionfilewriter = csv.writer(file)
                interactionfilewriter.writerow(
                    [
                        "agent_type",
                        "subject_ID",
                        "modelrun_ID",
                        "game_name",
                        "game_level",
                        "episode_number",
                        "event_name",
                        "count",
                    ]
                )

    # FOR Gym API
    def set_level(self, intended_level, intended_steps):
        self.Env.lvl = intended_level
        self.Env.set_level(self.Env.lvl)
        self.steps = intended_steps

    def get_level(self):
        return self.Env.lvl

    def step(self, action):
        if self.steps >= 1000000:
            sys.exit()

        self.steps += 1
        self.episode_steps += 1
        self.append_gif()
        if self.steps % 5000 == 0:
            self.save_gif()
            self.screen_history = []

        self.reward, self.game_over, self.win = self.Env.step(action)
        if len(self.Env.current_env._game.sprite_groups["avatar"]) > 0:
            self.avatar_position_data["data"].append(
                (
                    self.Env.current_env._game.sprite_groups["avatar"][0].rect.left,
                    self.Env.current_env._game.sprite_groups["avatar"][0].rect.top,
                    self.Env.current_env._game.time,
                    self.Env.lvl,
                    self.steps,
                )
            )
        else:
            print("AVATAR_ERROR_IGNORE")
            self.game_over = True
        
        objects = self.Env.get_objects()
        self.objects_position_data["data"].append(
            {
                "time": self.Env.current_env._game.time,
                "level": self.Env.lvl,
                "timestep": self.steps,
                "objects": objects,
            }
        )

        # PEDRO: 2. Store events that occur at each timestep
        timestep_events = set()
        for e in self.Env.current_env._game.effectListByClass:
            # because event handling is so weird in Frogs, we need to filter out these events.
            # Avatar-water and avatar-log collisions will still be reported from the (killSprite avatar water) interaction and (pullWithIt avatar log) interaction
            # which is what a player perceives when they play
            if e in [
                ("changeResource", "avatar", "water"),
                ("changeResource", "avatar", "log"),
            ]:
                pass
            else:
                timestep_events.add(tuple(sorted((e[1], e[2]))))

        for e in timestep_events:
            self.event_dict[e] += 1

        self.episode_reward += self.reward
        # self.reward = max(-1.0, min(self.reward, 1.0))
        # self.last_screen = self.current_screen
        self.state = self.get_screen()

        if self.game_over or self.episode_steps > self.timeout:
            if self.episode_steps > self.timeout:
                print("Game Timed Out")

            # PEDRO: 3. At the end of each episode, write events to csv
            if self.record_flag:
                with open(
                    "{}/{}_{}_object_interaction_history_{}.csv".format(
                        self.experiment_outpath, 
                        self.experiment_uuid,
                        self.game_name_short, 
                        self.level_switch
                    ),
                    "a",
                ) as file:
                    interactionfilewriter = csv.writer(file)
                    for event_name, count in self.event_dict.items():
                        row = (
                            "DDQN",
                            "NA",
                            "NA",
                            self.game_name_short,
                            self.Env.lvl,
                            self.episode,
                            event_name,
                            count,
                        )
                        interactionfilewriter.writerow(row)
            self.episode += 1
            print(
                "Level {}, episode reward at step {}: {}".format(
                    self.Env.lvl, self.steps, self.episode_reward
                )
            )
            sys.stdout.flush()
            episode_results = [
                self.Env.lvl,
                self.steps,
                self.episode_reward,
                self.win,
                self.game_name_short,
                int(self.criteria.split("/")[0]),
            ]

            self.recent_history.insert(0, self.win)
            self.recent_history.pop()
            if self.level_step():
                if self.record_flag:
                    with open(
                        "{}/{}_{}_reward_history_{}.csv".format(
                            self.experiment_outpath,
                            self.experiment_uuid,
                            self.game_name_short,
                            self.level_switch,
                        ),
                        "a",
                    ) as file:
                        writer = csv.writer(file)
                        writer.writerow(episode_results)
                    print("{}".format(1))
                    return self.state, self.reward, self.game_over, 0
            self.episode_reward = 0

            if self.record_flag:
                with open(self.avatar_file_path + f'_{self.episode}.p', "wb") as f:
                    cloudpickle.dump(self.avatar_position_data, f)

                with open(self.objects_file_path + f'_{self.episode}.p', "wb") as f:
                    cloudpickle.dump(self.objects_position_data, f)

                self.avatar_position_data['data'] = []
                self.objects_position_data['data'] = []

                with open(
                    "{}/{}_{}_reward_history_{}.csv".format(
                        self.experiment_outpath, 
                        self.experiment_uuid,
                        self.game_name_short, 
                        self.level_switch
                    ),
                    "a",
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(episode_results)
            #self.screen_history = []
        return self.state, self.reward, self.game_over, 0

    def reset(self):
        self.Env.reset()
        self.episode_steps = 0
        # self.last_screen = self.get_screen()
        # self.current_screen = self.get_screen()
        # self.state = current_screen - last_screen
        self.state = self.get_screen()
        return self.state

    # Screen functions from player.py
    def save_screen(self):
        misc.imsave("original.png", self.Env.render())
        misc.imsave(
            "altered.png", np.rollaxis(self.get_screen().cpu().numpy()[0], 0, 3)
        )

    def get_screen(self):
        # imageio.imsave('sample.png', self.Env.render())
        screen = self.Env.render()
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = cv2.resize(screen, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
        screen_1channel = np.mean(screen, axis=2)
        # return screen
        return screen_1channel

    def save_gif(self):
        imageio.mimsave(
            "{}/screens/{}_{}_frame_{}.gif".format(self.experiment_outpath, self.experiment_uuid, self.game_name_short, self.steps),
            self.screen_history,
        )

    def append_gif(self):
        frame = self.Env.render(gif=True)
        self.screen_history.append(frame)

    # Auxiliary functions from player.py
    def level_step(self):
        if self.level_switch == "sequential":
            if sum(self.recent_history) == int(
                self.criteria.split("/")[0]
            ):  # if level is 'won'
                if (
                    self.Env.lvl == len(self.Env.env_list) - 1
                ):  # if this is the last training level
                    print("Learning Finished")
                    return 1
                else:  # if this isn't the last level
                    self.Env.lvl += 1
                    self.Env.set_level(self.Env.lvl)
                    print("Next Level!")
                    self.recent_history = [0] * int(self.criteria.split("/")[1])
                    return 0

        # ANDRES Note that nothing happens otherwise
        elif self.level_switch == "random":
            # else:
            self.Env.lvl = np.random.choice(range(len(self.Env.env_list) - 1))
            self.Env.set_level(self.Env.lvl)
            return 0
        else:
            raise Exception("level switch not specified.")

    def close(self):
        pass
