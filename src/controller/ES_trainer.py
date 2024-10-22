import numpy as np
np.random.seed(2020)
from multiprocessing import Pool
from tqdm import tqdm
import time
from controller.Controller import *
from settings import *
from controller.ES_abstract import ES_abstract

class ES_trainer(ES_abstract):
    
    def __init__(self, loss_function, pop_size, elite_size, args):
        """
        Gaussian Evolution Strategy algorithm (without covariances)
        :param loss_function (function) rollout of the Neurosmash environment, function to be optimized
        :param pop_size   (float) population size
        :param elite_size (float) elite group size
        :param args
        """
        super().__init__(loss_function, pop_size, args)
        self.elite_size = elite_size
        self.weights = np.random.normal(0,1,self.dim)
        self.sigma = 1

    def train(self, n_iter, parallel=False, verbose=False):
        """
        :param n_iter    (int) number of iterations
        :param parallel (bool) If true, run the Neurosmash environment in parallel


        :return trained controller (Controller) trained controller object with trained weights
        :return fitness  (matrix) fitness score for each member for each iteration
        """
        pop_size, elite_size = self.pop_size, self.elite_size
        w, sigma  = self.weights, self.sigma
        reward = np.zeros((n_iter, 4)) # score of i) mean weight ii) best performer iii) worst performer iv) sampled population average
 
        tic = time.perf_counter()
        for i in tqdm(range(n_iter)):
            # Generate K population members
            population = np.random.multivariate_normal(mean=w, cov=sigma*np.eye(self.dim), size=pop_size)
            controllers = [Controller(self.args) for w in population]
            for c,w in zip(controllers,population):
                c.set_weight_array(w)
            # Multiprocess each population member
            if parallel:
                with Pool(os.cpu_count()) as pool:
                    fitness = pool.map(self.loss_func, controllers) # use starmap() for multiple argument functions.
                reward[i] = self.get_reward_stats(w, fitness)

            # or sequential training
            else:
                fitness = np.zeros(pop_size)
                for j in tqdm(range(pop_size)):
                    fitness[j] = self.loss_func(controllers[j], r_rounds=2)
                reward[i] = self.get_reward_stats(w, fitness)

            # Sort population and take elite            
            elite_idx  = np.argsort(-fitness)[:elite_size]
            elite      = population[elite_idx]

            # Update w and Σ using elite
            w = np.mean(elite, axis=0)
            sigma = np.sum((elite-w)**2) / elite_size

            if verbose:
                print(f'Mean reward: {reward[i, 0]}')
                print(f'Best reward: {reward[i, 1]}')
                print(f'Worst reward: {reward[i, 2]}')
                print(f'Sampled population reward: {reward[i, 3]}')

        toc = time.perf_counter()
        print(f'Duration of training the controller: {toc - tic:0.4f} seconds')
        
        self.weights = w
        self.sigma = sigma
        trained_controller = Controller(self.args)
        trained_controller.set_weight_array(w)
        return trained_controller, reward
