# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 20:35:06 2021

@author: 123
"""
import gym
from gym_super_mario_bros.actions import RIGHT_ONLY
import gym_super_mario_bros
from gym import Wrapper
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY,COMPLEX_MOVEMENT
from gym.wrappers import FrameStack
import numpy as np
import torch
from torchvision import transforms as T

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)

        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            
            if(reward<0):
                if(action in range(6,10) and done is False):
                   reward/=10
            
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    

    
class RewardFunction(Wrapper):
    def __init__(self, env=None):
        super(RewardFunction, self).__init__(env)
        self.curr_score = 0
        self.max_x=0
    def step(self, action):
        state, reward, done, info = self.env.step(action)
       
        
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 350
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0

        return (self.env.reset())
    
"""
class RewardFunction(Wrapper):
    def __init__(self, env=None):
        super(RewardFunction, self).__init__(env)
        self.curr_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        return (self.env.reset())
"""
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        observation=observation.view(1,1,80,80)
        return observation
    
def create_env(world,stage):
    env = gym_super_mario_bros.make( 'SuperMarioBros-{}-{}-v0'.format(world,stage) )
    env = JoypadSpace ( env , COMPLEX_MOVEMENT )
    env=  RewardFunction(env)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=80)
    #env = FrameStack(env, num_stack=4)
    return env
    