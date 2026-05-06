import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import gymnasium as gym
import evogym.envs
import NeuralEvolution
import Utilities

def initialize_empty():
    
    body = np.zeros((5,5), dtype=int)

    return body


def initialize_random():
    
    body = np.zeros((5,5), dtype=int)

    for i in range(5):
        for j in range(5):
             body[i][j] = np.random.randint(0,5)

    return body


def initialize_default():
    
    body = np.zeros((5,5), dtype=int)
    body[1:4,1:4] = 2

    return body

def initialize_default_full():
    
    body = 2*np.ones((5,5), dtype=int)

    return body

def initialize_muscle_body():
    
    body = np.zeros((5,5), dtype=int)
    body[1:4,1:4] = 4

    return body

def initialize_muscle_body_full():
    
    body = 4*np.ones((5,5), dtype=int)

    return body

def initialize_chess_body():
    
    body = np.zeros((5,5), dtype=int)

    for row in range(5):
        for col in range(5):
            if ((row+col)%2) == 0:
                body[row][col] = 3
            else:
                body[row][col] = 4

    return body

def initialize_horse():
    
    body = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [4, 4, 0, 4, 4],
    [4, 4, 0, 4, 4],
    [4, 4, 0, 4, 4]
    ])

    return body

def mutate_body(body, mutation_amount = 5, forceType = False, c = 0, show = False):
        
        mutated_body = np.copy(body)
        
        for _ in range(mutation_amount):

            i = np.random.randint(0,5)
            j = np.random.randint(0,5)

            if forceType:
                k = c
            else:
                current = mutated_body[i][j]
                allowed_values = np.setdiff1d(range(0, 5), current)
                k = np.random.choice(allowed_values)

            willBreak = unicity_check(mutated_body,i,j,k)

            if willBreak:
                mutated_body = mutate_body(mutated_body, mutation_amount = 1, forceType = True, c = k)
            else:
                mutated_body[i][j] = k

        if((3 not in mutated_body) and (4 not in mutated_body)):
            mutated_body[2][2] = 3

        if show:
            print(mutated_body)
        return mutated_body

def unicity_check(mutated_body,a,b,c):

    possible_body = np.copy(mutated_body)
    possible_body[a][b] = c

    willBreak = False
    blocks = 0
    visited = np.zeros((5,5), dtype=bool)
    visited = possible_body == 0

    def explore(i,j):
        if i < 0 or i >= 5 or j < 0 or j >= 5 or visited[i][j]:
            return
        visited[i][j] = True
        explore(i,j+1)
        explore(i,j-1)
        explore(i+1,j)
        explore(i-1,j)

    for r in range(5):
        for c in range(5):
            if not visited[r][c]:
                explore(r,c)
                blocks += 1

    if blocks != 1:
        willBreak = True

    return willBreak

def body_sewing(new_body):

    blocks = 0
    visited = np.zeros((5,5), dtype=bool)
    visited = new_body == 0

    found = False

    for rz in range(5):
        if found == True:
            break
        for cz in range(5):
            if visited[rz][cz]:
                void_x = rz
                void_y = cz
                # print("("+str(void_x)+","+str(void_y)+")")
                found = True
                break

    def explore(i,j):
        if i < 0 or i >= 5 or j < 0 or j >= 5 or visited[i][j]:
            return
        visited[i][j] = True
        explore(i,j+1)
        explore(i,j-1)
        explore(i+1,j)
        explore(i-1,j)

    for r in range(5):
        for c in range(5):
            if not visited[r][c]:
                explore(r,c)
                blocks += 1

    if blocks != 1:
        new_body[void_x][void_y] = 2
        new_body = body_sewing(new_body)

    return new_body

def one_plus_lambda(setup, save_graphs = True, save_anims = False, save_solutions = True, old_fitness = -np.Inf, old_total_eval = 0):
    
    # Elite
    elite_body = setup["body"]
    cur_body_fit = old_fitness
    cycles = setup["cycles"]
    amount = setup["amount"]

    map = setup["map"]
    mutations = setup["mutations"]
    alpha = setup["alpha"]

    global_fit = []
    global_eval = []

    cur_total_eval = old_total_eval

    for cycle in range(cycles):

        print("GENERATION "+str(cycle+1))

        test_bodies = []
        body_fitness = []

        if save_graphs:
            brains_evals = []
            brains_fits = []

        if save_anims or save_solutions:
            test_brains = []
        
        for i in range(amount):
            test_bd = mutate_body(elite_body, mutation_amount = mutations[map[cycle]], forceType=False, show=False)
            test_bodies.append(test_bd)

        test_body_counter = 0

        for test_bd in test_bodies:

            test_body_counter += 1
            print("Body "+str(test_body_counter))
            print(test_bd)

            setup["body"] = test_bd

            evolved_br, eval_history, fit_history = NeuralEvolution.weighted_ES(setup,cycle)

            body_fitness.append(evolved_br.fitness)

            if save_graphs:
                brains_evals.append(eval_history)
                brains_fits.append(fit_history)

            if save_anims or save_solutions:
                test_brains.append(evolved_br)

        better_bodies = np.zeros(amount, dtype=bool)
        for i in range(amount):
            better_bodies[i] = body_fitness[i] > cur_body_fit
        beta = np.sum(better_bodies)

        # sort by fitness
        inv_body_fitness = [- f for f in body_fitness]
        # indices from highest fitness to lowest
        idx = np.argsort(inv_body_fitness)

        mutation_stack = np.zeros((5,5), dtype=int)

        cur_alpha = alpha[map[cycle]]

        for i in range(np.min([cur_alpha,beta])):

            mutation = test_bodies[idx[cur_alpha-1-i]]-elite_body
            mask = mutation != 0
            mutation_stack[mask] = mutation[mask]

        elite_body += mutation_stack

        elite_body = body_sewing(elite_body)

        if beta > 0:
            cur_body_fit = body_fitness[idx[0]]

            if save_graphs:
                Utilities.generate_graph(setup, brains_evals[idx[0]], brains_fits[idx[0]], "Generation_"+str(cycle+1))

            if save_anims:
                setup["body"] = test_bodies[idx[0]]
                Utilities.generate_GIF(test_brains[idx[0]], setup, "Generation_"+str(cycle+1))

            if save_solutions:
                setup["body"] = test_bodies[idx[0]]
                Utilities.save_solution(test_brains[idx[0]], setup, "Generation_"+str(cycle+1))

        cur_gen = setup["generations"]
        cur_gen = cur_gen[map[cycle]]
        cur_lambda = setup["lambda"]
        cur_lambda = cur_lambda[map[cycle]]

        cur_total_eval += amount*cur_gen*cur_lambda

        global_fit.append(cur_body_fit)
        global_eval.append(cur_total_eval)
        
        Utilities.generate_graph(setup, global_eval, global_fit, "Global_Result")

    print("PROCESS FINISHED")