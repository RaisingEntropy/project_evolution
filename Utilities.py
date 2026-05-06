import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import gymnasium as gym
from evogym.utils import get_full_connectivity
from tqdm import tqdm
import json
import imageio
import NeuralNetwork
import Environment

def initial_warning(setup):
    setup["cycles"] = len(setup["map"])
    gen_count = setup["generations"]
    lambda_count = setup["lambda"]
    total_steps = []

    for i in range(len(gen_count)):
        total_steps.append(gen_count[i]*lambda_count[i]*setup["max_steps"])
        
    max_steps = np.max(total_steps)

    print("Max total steps: "+str(max_steps))
    if(max_steps > 5000000):
        print("WARNING! One or more simulations exceed the maximum allowed amount of steps !")

def save_solution(ind, setup, name="solution"):
    dim = Environment.get_dim(setup["env_name"], robot=setup["body"]) # Get network dims
    config = {**setup, **dim} # Merge configs
    save_cfg = {}
    for i in ["env_name", "body", "n_in", "h_size", "n_out"]:
        assert i in config, f"{i} not in config"
        save_cfg[i] = config[i]
    save_cfg["body"] = config["body"].tolist()
    save_cfg["genes"] = ind.genes.tolist()
    save_cfg["fitness"] = float(ind.fitness)
    # save
    name = setup["folder"]+name+".json"
    with open(name, "w") as f:
        json.dump(save_cfg, f)
    return save_cfg

def load_solution(name="solution.json"):
    with open(name, "r") as f:
        cfg = json.load(f)
    cfg["body"] = np.array(cfg["body"])
    cfg["genes"] = np.array(cfg["genes"])
    ind = NeuralNetwork.Agent(NeuralNetwork.Network, cfg, genes=cfg["genes"])
    ind.fitness = cfg["fitness"]
    setup = {
        "env_name": cfg["env_name"],
        "body": cfg["body"]
    }
    return ind, setup

def generate_graph(setup, total_evals, fits, robot_name="Robot"):
    total_evals = np.asarray(total_evals, dtype=np.float32)
    fits = np.asarray(fits, dtype=np.float32)
    support_array = np.vstack((total_evals,fits))
    support_file = robot_name
    support_file = setup["folder"]+support_file+"_data.txt"
    with open(support_file, "w") as txt_file:
        for line in support_array:
            line = str(line)
            txt_file.write(" ".join(line) + "\n") # works with any number of elements in a line

    plt.plot(total_evals, fits)
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    robot_name = setup["folder"]+robot_name+".png"
    plt.savefig(fname = robot_name,format="png")
    plt.clf()

def generate_GIF(ind=None, setup=None, robot_name="Robot"):
    dim = Environment.get_dim(setup["env_name"], robot=setup["body"]) # Get network dims
    config = {**setup, **dim} # Merge configs

    env = Environment.make_env(config["env_name"], robot=config["body"], render_mode="rgb_array")
    env.metadata.update({'render_modes': ["rgb_array"]})
    env.metadata.update({'render_fps': 60})

    ind.fitness, imgs = Environment.evaluate(ind, env, render=True)
    env.close()
    robot_name = setup["folder"]+robot_name+".gif"
    imageio.mimsave(robot_name, imgs, duration=(1/50.0))