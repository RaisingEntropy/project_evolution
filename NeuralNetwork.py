import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Network(nn.Module):
    def __init__(self, n_in, h_size, n_out): #Initialization of the neural network structure by PyTorch
        super().__init__()
        self.fc1 = nn.Linear(n_in, h_size) #Layer 1
        self.fc2 = nn.Linear(h_size, h_size) #Layer 2
        self.fc3 = nn.Linear(h_size, n_out) #Layer 3
 
        self.n_out = n_out

    def reset(self):
        pass
    
    def forward(self, x):
        x = self.fc1(x) #Push information through layer 1
        x = F.relu(x) #Rectify the output for negatives

        x = self.fc2(x) #Push information through layer 2
        x = F.relu(x) #Rectify the output for negatives

        x = self.fc3(x) #Push information through layer 3
        return x #Return information
    
class Agent:
    def __init__(self, Net, config, genes = None):
        self.config = config #Configs for network topology
        self.Net = Net #Network attribute initialization
        self.model = None #Model initialization
        self.fitness = None #Fitness initialization

        self.device = torch.device("cpu")

        self.make_network()
        if genes is not None:
            self.genes = genes

    def __repr__(self):  # pragma: no cover
        return f"Agent {self.model} > fitness={self.fitness}"

    def __str__(self):  # pragma: no cover
        return self.__repr__()

    def make_network(self): #Method for creating the neural network object and saving its corresponding model
        n_in = self.config["n_in"]
        h_size = self.config["h_size"]
        n_out = self.config["n_out"]
        self.model = self.Net(n_in, h_size, n_out).to(self.device).double()
        return self

    @property
    def genes(self): #Method for returning the parameters of the model as a genes vector
        if self.model is None:
            return None
        with torch.no_grad():
            params = self.model.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().double().numpy()

    @genes.setter
    def genes(self, params): #Method for setting the parameters of the model from a genes vector and resetting the model and the fitness
        if self.model is None:
            self.make_network()
        assert len(params) == len(self.genes), "Genome size does not fit the network size"
        if np.isnan(params).any():
            raise
        a = torch.tensor(params, device=self.device)
        torch.nn.utils.vector_to_parameters(a, self.model.parameters())
        self.model = self.model.to(self.device).double()
        self.fitness = None
        return self

    def mutate_ga(self): #Method for mutating the genome of an agent, replaces old genes with noise at random
        genes = self.genes
        n = len(genes)
        f = np.random.choice([False, True], size=n, p=[1/n, 1-1/n])
        
        new_genes = np.empty(n)
        new_genes[f] = genes[f]
        noise = np.random.randn(n-sum(f))
        new_genes[~f] = noise
        return new_genes

    def act(self, obs):
        # continuous actions
        with torch.no_grad():
            x = torch.tensor(obs).double().unsqueeze(0).to(self.device)
            actions = self.model(x).cpu().detach().numpy()
        return actions

