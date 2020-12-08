import numpy as np
np.random.seed(2020)
from multiprocessing import Pool
import os
from tqdm import tqdm
import time
from settings import *
from rollout import rollout
from controller.Controller import *

class ES_trainer():
    
    def __init__(self, Λ, λ):
        """
        Gaussian Evolution Strategy algorithm (without covariances)
        :param Λ         (float) population size
        :param λ         (float) elite group size 
        :param w_dim     (int) shape of weights to be optimized
        """
        self.λ = λ
        self.Λ = Λ
        self.w_dim = z_dim + h_dim
        self.weights = np.random.normal(0,1,self.w_dim)
        self.σ = 1
        self.loss_func = rollout
        
    def train(self, n_iter):
        """
        :param n_iter    (int) number of iterations

        :return θ        (matrix) parameter values for each iteration, 
        :return fitness  (matrix) fitness score for each member for each iteration
        """
        Λ, λ   = self.Λ, self.λ
        w, σ   = self.weights, self.σ
        reward = np.zeros((n_iter, Λ+1))
 
        tic = time.perf_counter()
        for i in tqdm(range(n_iter)):
            # Generate λ population members
            population = np.random.multivariate_normal(mean=w, cov=σ*np.eye(self.w_dim), size=Λ)
            population = [Controller(weights) for weights in population]
            
            # Multiprocess each population member
            with Pool(os.cpu_count()) as pool:
                fitness = pool.map(self.loss_func, population) # use starmap() for multiple argument functions.
            reward[i,:] = np.append(self.loss_func(w), fitness)

            # Sort population and take elite            
            elite_idx  = np.argsort(fitness)[:λ]
            elite      = population[elite_idx]

            # Update w and Σ using elite
            w = np.mean(elite, axis=0)
            σ = np.sum((elite-w)**2) / λ


        toc = time.perf_counter()
        print(f'Time elapsed: {toc - tic:0.4f} seconds')
        
        self.weights = w
        self.σ = σ
        return w, reward
    
