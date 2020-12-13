import numpy as np
np.random.seed(2020)
from multiprocessing import Pool
from tqdm import tqdm
import time
from controller.Controller import *
import matplotlib.pyplot as plt

class ES_trainer():
    
    def __init__(self, loss_function, pop_size, elite_size, args):
        """
        Gaussian Evolution Strategy algorithm (without covariances)
        :param pop_size   (float) population size
        :param elite_size (float) elite group size
        :param w_dim      (int) shape of weights to be optimized
        """
        self.args = args
        self.elite_size = elite_size
        self.pop_size = pop_size
        self.w_dim = z_dim + h_dim
        self.weights = np.random.normal(0,1,self.w_dim)
        self.sigma = 1
        self.loss_func = loss_function
        
    def train(self, n_iter, parallel=False):
        """
        :param n_iter    (int) number of iterations

        :return trained controller (Controller) trained controller object with trained weights
        :return fitness  (matrix) fitness score for each member for each iteration
        """
        pop_size, elite_size = self.pop_size, self.elite_size
        w, sigma  = self.weights, self.sigma
        reward = np.zeros((n_iter, 4)) # score of i) mean weight ii) best performer iii) worst performer iv) sampled population average
 
        tic = time.perf_counter()
        for i in tqdm(range(n_iter)):
            # Generate K population members
            population = np.random.multivariate_normal(mean=w, cov=sigma*np.eye(self.w_dim), size=pop_size)
            controllers = [Controller(self.args, weights) for weights in population]
            
            # Multiprocess each population member
            if parallel:
                with Pool(os.cpu_count()) as pool:
                    fitness = pool.map(self.loss_func, controllers) # use starmap() for multiple argument functions.
                reward[i] = self.get_reward_stats(w, fitness)

            # or sequential training
            else:
                fitness = np.zeros(pop_size)
                for j in tqdm(range(pop_size)):
                    fitness[j] = self.loss_func(controllers[j])
                reward[i] = self.get_reward_stats(w, fitness)

            # Sort population and take elite            
            elite_idx  = np.argsort(fitness)[:elite_size]
            elite      = population[elite_idx]

            # Update w and Σ using elite
            w = np.mean(elite, axis=0)
            sigma = np.sum((elite-w)**2) / elite_size


        toc = time.perf_counter()
        print(f'Duration of training the controller: {toc - tic:0.4f} seconds')
        
        self.weights = w
        self.sigma = sigma
        return Controller(self.args, w), reward

    def plot_results(self, reward):
        plt.figure(figsize=(8,5))
        plt.title(f'Cumulative reward of as a function generations')
        plt.xlabel('Generation')
        plt.ylabel('Cumulative reward')

        # mean population
        plt.plot(np.arange(reward.shape[0]), reward[:,0], label="Population solution")
        # best performer
        plt.plot(np.arange(reward.shape[0]), reward[:,1], label="Best performer")
        # Worst performer
        plt.plot(np.arange(reward.shape[0]), reward[:,2], label="Worst performer")
        # sampled population average
        plt.plot(np.arange(reward.shape[0]), reward[:,3], label="Sampled population reward")

        plt.legend()
        plt.savefig('data/images/ControllerResults/controller_training.pdf', dpi=200)
        plt.show()

    def get_reward_stats(self, w, fitness):
        reward = np.zeros(4)
        reward[0] = self.loss_func(Controller(self.args, w))  # First entry is of mean weights, after that of pop. members
        reward[1] = np.max(fitness)  # best performer
        reward[2] = np.min(fitness)  # worst performer
        reward[3] = np.mean(fitness)  # sampled population mean
        return reward

    
