import numpy as np

np.random.seed(2020)
from multiprocessing import Pool
from tqdm import tqdm
import time
from utils.rollout import RolloutGenerator
from controller.Controller import *


class ES_trainer():

    def __init__(self, rollout_generator, pop_size, elite_size):
        """
        Gaussian Evolution Strategy algorithm (without covariances)
        :param pop_size   (float) population size
        :param elite_size (float) elite group size
        :param w_dim      (int) shape of weights to be optimized
        """
        self.elite_size = elite_size
        self.pop_size = pop_size
        self.w_dim = z_dim + h_dim
        self.weights = np.random.normal(0, 1, self.w_dim)
        self.sigma = 1
        self.loss_func = rollout_generator.rollout

    def train(self, n_iter, parallel=False):
        """
        :param n_iter    (int) number of iterations
        :return trained controller (Controller) trained controller object with trained weights
        :return fitness  (matrix) fitness score for each member for each iteration
        """
        pop_size, elite_size = self.pop_size, self.elite_size
        w, sigma = self.weights, self.sigma
        reward = np.zeros((n_iter, pop_size + 1))

        tic = time.perf_counter()
        for i in tqdm(range(n_iter)):
            # Generate K population members
            population = np.random.multivariate_normal(mean=w, cov=sigma * np.eye(self.w_dim), size=pop_size)
            controllers = [Controller(weights) for weights in population]

            # Multiprocess each population member
            if parallel:
                with Pool(os.cpu_count()) as pool:
                    fitness = pool.map(self.loss_func, controllers)  # use starmap() for multiple argument functions.
                reward[i, :] = np.append(self.loss_func(Controller(w)),
                                         fitness)  # First entry is of mean weights, after that of pop. members
            # or sequential training
            else:
                fitness = np.zeros(pop_size)
                for i in range(pop_size):
                    fitness[i] = self.loss_func(controllers[i])
                reward[i, :] = np.append(self.loss_func(Controller(w)), fitness)

            # Sort population and take elite
            elite_idx = np.argsort(fitness)[:elite_size]
            elite = population[elite_idx]

            # Update w and Σ using elite
            w = np.mean(elite, axis=0)
            sigma = np.sum((elite - w) ** 2) / elite_size

        toc = time.perf_counter()
        print(f'Duration of training the controller: {toc - tic:0.4f} seconds')

        self.weights = w
        self.sigma = sigma
        return Controller(w), reward