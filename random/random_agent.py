import imageio  
import cv2
import csv
import gym
import numpy as np
from tqdm import tqdm
import time
import uuid 
import os


def log_game_score(
    path,
    experiment_id,
    experiment_timestep,
    episode,
    episode_score,
    timestep_episode,
    best_score,
    accumulated_score,
):

    filename = "{}/game_score_{}.csv".format(path, experiment_id)

    # We have human max score data reported every 15 seconds.
    if not os.path.exists(filename):

        cols = [
            "experiment_timestep",
            "episode",
            "episode_score",
            "timestep_episode",
            "best_score",
            "accumulated_score",
        ]
        with open(filename, "w") as f:
            csv.writer(f).writerow(cols)

    vals = [
        experiment_timestep,
        episode,
        episode_score,
        timestep_episode,
        best_score,
        accumulated_score,
    ]
    with open(filename, "a") as f:
        csv.writer(f).writerow(vals)



class AtariPreprocessing(object):
  """A class implementing image preprocessing for Atari 2600 agents.

  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).
    * Grayscale and max-pooling of the last two frames.
    * Downsample the screen to a square image (defaults to 84x84).

  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".
  """

  def __init__(self, environment, frame_skip=4, terminal_on_life_loss=False,
               screen_size=84, game_name='', parameter_set='', tag='', log_video=False, output_name = 'random_policy'):
    """Constructor for an Atari 2600 preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.
      frame_skip: int, the frequency at which the agent experiences the game.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, size of a resized Atari 2600 frame.

    Raises:
      ValueError: if frame_skip or screen_size are not strictly positive.
    """
    if frame_skip <= 0:
      raise ValueError('Frame skip should be strictly positive, got {}'.
                       format(frame_skip))
    if screen_size <= 0:
      raise ValueError('Target screen size should be strictly positive, got {}'.
                       format(screen_size))

    date = time.strftime("%Y.%m.%d")
    self.experiment_uuid = uuid.uuid1()

    self.experiment_id = game_name + '_' + str(tag)
    self.experiment_outpath = "experiments/{}/{}".format(
       output_name, game_name
    )

    os.makedirs(self.experiment_outpath, exist_ok=True)

    self.log_video = log_video
    self.imgs = []
    self.environment = environment
    self.terminal_on_life_loss = terminal_on_life_loss
    self.frame_skip = frame_skip
    self.screen_size = screen_size

    self.game_name = game_name
    self.experiment_timestep = 0 
    self.episode_timestep = 0
    self.episode = 0
    self.episode_score = 0
    self.best_score = 0
    self.accumulated_score = 0

    obs_dims = self.environment.observation_space
    # Stores temporary observations used for pooling over two successive
    # frames.
    self.screen_buffer = [
        np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
        np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
    ]

    self.game_over = False
    self.lives = 0  # Will need to be set by reset().

  @property
  def observation_space(self):
    # Return the observation space adjusted to match the shape of the processed
    # observations.
    return Box(low=0, high=255, shape=(self.screen_size, self.screen_size, 1),
               dtype=np.uint8)

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def reward_range(self):
    return self.environment.reward_range

  @property
  def metadata(self):
    return self.environment.metadata

  def close(self):
    return self.environment.close()

  def reset(self):

    self.environment.reset()
    self.lives = self.environment.ale.lives()
    self.screen_buffer[1].fill(0)
    return self._pool_and_resize()


  def render(self, mode):
    """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
    return self.environment.render(mode)


  def step(self, action):

    accumulated_reward = 0.
    
    for time_step in range(self.frame_skip):
      # We bypass the Gym observation altogether and directly fetch the
      # grayscale image from the ALE. This is a little faster.
      _, reward, game_over, info = self.environment.step(action)
      accumulated_reward += reward

      self.accumulated_score += reward
      self.experiment_timestep += 1
      self.episode_timestep += 1

      if (self.game_name == "Pong" and reward > 0) or (self.game_name != "Pong"):
        self.episode_score += reward
      
      self.best_score = max(self.best_score, self.episode_score)

      if self.log_video:
        self.imgs.append(self.render('rgb_array'))

      # Log every 15 seconds
      if self.experiment_timestep % 900 == 0 or game_over:

        if game_over and self.log_video:
          filename = "{}/screen_{}.gif".format(self.experiment_outpath, self.experiment_timestep)

          imageio.mimsave(
            filename,
            self.imgs
          )
          self.imgs = []

        log_game_score(
                self.experiment_outpath,
                self.experiment_id,
                self.experiment_timestep,
                self.episode,
                self.episode_score,
                self.episode_timestep,
                self.best_score,
                self.accumulated_score,
        )

      if self.terminal_on_life_loss:
        new_lives = self.environment.ale.lives()
        is_terminal = game_over or new_lives < self.lives
        self.lives = new_lives
      else:
        is_terminal = game_over

      if game_over:
        self.episode += 1
        self.episode_timestep = 0
        self.episode_score = 0
        break
      # We max-pool over the last two frames, in grayscale.
      elif time_step >= self.frame_skip - 2:
        t = time_step - (self.frame_skip - 2)
        self._fetch_grayscale_observation(self.screen_buffer[t])

    # Pool the last two observations.
    observation = self._pool_and_resize()

    self.game_over = game_over
    return observation, accumulated_reward, is_terminal, info

  def _fetch_grayscale_observation(self, output):
    """Returns the current observation in grayscale.

    The returned observation is stored in 'output'.

    Args:
    output: numpy array, screen buffer to hold the returned observation.

    Returns:
    observation: numpy array, the current observation in grayscale.
    """
    self.environment.ale.getScreenGrayscale(output)
    return output

  def _pool_and_resize(self):
    """Transforms two frames into a Nature DQN observation.

    For efficiency, the transformation is done in-place in self.screen_buffer.

    Returns:
    transformed_screen: numpy array, pooled, resized screen.
    """
    # Pool if there are enough screens to do so.
    if self.frame_skip > 1:
      np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                out=self.screen_buffer[0])

    transformed_image = cv2.resize(self.screen_buffer[0],
                                (self.screen_size, self.screen_size),
                                interpolation=cv2.INTER_AREA)
    int_image = np.asarray(transformed_image, dtype=np.uint8)
    return np.expand_dims(int_image, axis=2)


