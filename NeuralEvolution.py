import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import gymnasium as gym
import evogym.envs
from evogym.utils import get_full_connectivity
from tqdm import tqdm
import Environment
import NeuralNetwork
from multiprocessing import Pool
import time

def weighted_ES(setup, cycle):

    xmap = setup["map"]
    xgenerations = setup["generations"]
    xlambda = setup["lambda"]

    dim = Environment.get_dim(setup["env_name"], robot=setup["body"]) # Get network dims
    config = {**setup, **dim} # Merge configs
    
    # Update weights
    mu = config["mu"]
    w = np.array([np.log(mu + 0.5) - np.log(i) for i in range(1, mu + 1)])
    w /= np.sum(w)
    
    env = Environment.make_env(config["env_name"], robot=config["body"])

    # Center of the distribution
    elite = NeuralNetwork.Agent(NeuralNetwork.Network, config)

    elite.fitness = -np.inf
    theta = elite.genes
    d = len(theta)

    fit_history = []
    eval_history = []

    bar = tqdm(range(xgenerations[xmap[cycle]]))
    
    for gen in bar:
        population = []
        for i in range(xlambda[xmap[cycle]]):
            genes = theta + np.random.randn(len(theta)) * config["sigma"]
            ind = NeuralNetwork.Agent(NeuralNetwork.Network, config, genes=genes)
            # ind.fitness = evaluate(ind, env, max_steps=config["max_steps"])
            population.append(ind)

        with Pool(processes=len(population)) as pool:
            pop_fitness = pool.starmap(Environment.mp_eval, [(a, config) for a in population])
        
        #pop_fitness = [Environment.evaluate(ind, env, max_steps=config["max_steps"]) for ind in population]
        
        for i in range(len(population)):
            population[i].fitness = pop_fitness[i]

        # sort by fitness
        inv_fitnesses = [- f for f in pop_fitness]
        # indices from highest fitness to lowest
        idx = np.argsort(inv_fitnesses)
        
        step = np.zeros(d)
        for i in range(mu):
            # update step
            step = step + w[i] * (population[idx[i]].genes - theta)
        # update theta
        theta = theta + step * config["lr"]

        if pop_fitness[idx[0]] > elite.fitness:
            elite.genes = population[idx[0]].genes
            elite.fitness = pop_fitness[idx[0]]

        fit_history.append(elite.fitness)
        eval_history.append(len(population) * (gen+1))

        bar.set_description(f"Best: {elite.fitness}")
        
    env.close()
    
    return elite, eval_history, fit_history