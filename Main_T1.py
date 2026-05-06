import numpy as np
import BodyEvolution
import Utilities

def main():
    print("Hello from project!")

    initial_body = np.array([
    [0, 0, 1, 0, 3],
    [0, 3, 1, 0, 4],
    [0, 0, 2, 4, 4],
    [0, 2, 2, 0, 3],
    [0, 0, 0, 0, 0]
    ])

    print(initial_body)

    setup = {
        # Environment and robot definition;
        "env_name": "Thrower-v0",
        "body": initial_body,
        # Body evolution parameters;
        "amount": 4,
        "map" : [1,1,1,1,2,2,2],
        "mutations" : [5,4,3],
        "alpha": [2,1,1],
        # Brain evolution parameters;
        "generations": [40,80,400], # to change: increase!
        "lambda": [50,40,25], # Population size
        "mu": 5, # Parents pop size
        "sigma": 0.1, # mutation std
        "lr": 1.25, # Learning rate
        "max_steps": 500, # to change to 500
        "folder": "Thrower1/"
    }

    Utilities.initial_warning(setup)

    BodyEvolution.one_plus_lambda(setup, save_graphs = True, save_anims = True, save_solutions = True, old_fitness = 1.8143061, old_total_eval = 49600)


if __name__ == "__main__":
    main()