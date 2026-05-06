import numpy as np
import BodyEvolution
import Utilities

def main():
    print("Hello from project!")

    initial_body = BodyEvolution.initialize_chess_body()

    print(initial_body)

    setup = {
        # Environment and robot definition;
        "env_name": "Climber-v2",
        "body": initial_body,
        # Body evolution parameters;
        "amount": 4,
        "map" : [0,0,0,0,0,1,1,1,1,2,2,2],
        "mutations" : [6,4,3],
        "alpha": [2,1,1],
        # Brain evolution parameters;
        "generations": [60,120,400], # to change: increase!
        "lambda": [50,40,25], # Population size
        "mu": 5, # Parents pop size
        "sigma": 0.1, # mutation std
        "lr": 1.25, # Learning rate
        "max_steps": 500, # to change to 500
        "folder": "Climber3/",
    }

    Utilities.initial_warning(setup)

    BodyEvolution.one_plus_lambda(setup, save_graphs = True, save_anims = True, save_solutions = True)


if __name__ == "__main__":
    main()