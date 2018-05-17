from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import pandas as pd
from collections import deque

class Atari(object):

    def __init__(self, env, resize_height, resize_width, collate_length, game_name):
        self.game_env = env
        self.height = resize_height
        self.width = resize_width
        if 'Breakout' in game_name:
            self.n_actions = 3
        else:
            self.n_actions = self.game_env.action_space.n
        self.collate_frames = collate_length
        self.game_controls = list(range(1, self.n_actions+1))


    def get_initial_frames(self):
        self.X_t = self.game_env.reset()
        self.state_t = np.zeros([self.height, self.width, self.collate_frames])
        self.X_t_prime = self.resize_convert_frames(self.X_t)
        self.memory_states = []
        for ix in range(self.collate_frames):
            self.state_t[:, :, ix] = self.X_t_prime
            self.memory_states.append(self.X_t_prime)
        return self.state_t


    def resize_convert_frames(self, input_frames):
        return resize(rgb2gray(input_frames), (self.height, self.width))


    def step_in_game(self, action_index):
        new_X_t, self.r_t, self.isdone, self.info = self.game_env.step(self.game_controls[action_index])
        new_X_t = self.resize_convert_frames(new_X_t)
        self.current_state = []
        for s_t in self.memory_states[1:]:
            self.current_state.append(s_t)
        self.current_state.append(new_X_t)
        self.state_t = np.zeros([self.height, self.width, self.collate_frames])
        self.memory_states = []
        for ix, _ in enumerate(self.current_state):
            self.state_t[:, :, ix] = self.current_state[ix]
            self.memory_states.append(self.current_state[ix])
        return self.state_t, self.r_t, self.isdone, self.info

