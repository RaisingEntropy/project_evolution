from evogym import sample_robot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import gymnasium as gym
from evogym.utils import get_full_connectivity
from tqdm import tqdm

def make_env(env_name, robot=None, seed=None, **kwargs):
    if robot is None: 
        env = gym.make(env_name, **kwargs)
    else:
        connections = get_full_connectivity(robot)
        env = gym.make(env_name, body=robot, connections=connections, **kwargs)
    env.robot = robot
    if seed is not None:
        env.seed(seed)
        
    return env

def get_dim(env_name, robot=None):
    env = make_env(env_name, robot)
    dim = {
        "n_in": env.observation_space.shape[0],
        "h_size": 32,
        "n_out": env.action_space.shape[0],
    }
    env.close()
    return dim

def evaluate(agent, env, max_steps=500, render=False):
    obs, i = env.reset()
    agent.model.reset()
    reward = 0
    steps = 0
    done = False
    if render:
        imgs = []
    while not done and steps < max_steps:
        if render:
            img = env.render() #mode='img'
            imgs.append(img)
        action = agent.act(obs)
        obs, r, done, trunc,  _ = env.step(action)
        reward += r
        steps += 1
        
    if render:
        return reward, imgs
    return reward

def mp_eval(a, cfg):
    env = make_env(cfg["env_name"], robot=cfg["body"])
    fit = evaluate(a, env, max_steps=cfg["max_steps"])
    env.close()
    return fit