import numpy as np
np.random.seed(2020)
from multiprocessing import Pool
from tqdm import tqdm
import time
from controller.Controller import *
from settings import *
from controller.ES_trainer import ES_abstract


class NES_trainer(ES_abstract):

    def __init__(self, loss_function, pop_size, args, p_theta, learn_rate):
        """
        Natural Evolution Strategy algorithm
        :param loss_function (function) rollout of the Neurosmash environment, function to be optimized
        :param pop_size   (float) population size
        :param p_theta     (function) probability density function of theta
        :param learn_rate  (float) learning rate

        :param args
        """
        super().__init__(loss_function, pop_size, args)
        self.weights = np.random.normal(0, 1, self.dim) # weights = mu
        self.cov_matrix = np.eye(self.dim)
        #self.theta =
        self.grad = self.grad_log_mvn_pdf
        self.learn_rate = learn_rate
        self.p_theta = p_theta


    def train(self, n_iter, parallel=False):
        '''
        Natural Evolution Strategy algorithm
        :param n_iter    (int) number of iterations
        :param parallel (bool) If true, run the Neurosmash environment in parallel

        :return trained controller (Controller) trained controller object with trained weights
        :return fitness  (matrix) fitness score for each member for each iteration
        '''
        pop_size, learn_rate = self.pop_size, self.learn_rate
        p_theta, w = self.p_theta, self.weights
        grad = self.grad

        reward = np.zeros((n_iter,4))  # score of i) mean weight ii) best performer iii) worst performer iv) sampled population average

        tic = time.perf_counter()
        for i in tqdm(range(n_iter)):
            # Reset gradient and Fisher matrix
            grad_J = np.zeros((self.dim+np.sum(np.arange(self.dim+1)), 1))
            F = np.zeros((self.dim+np.sum(np.arange(self.dim+1)), 1))

            # Specify current pi( | theta) and current gradient
            cur_p_theta = p_theta(theta)
            cur_grad = grad(theta)

            # Gradient descent
            fitness = np.zeros(pop_size)
            for j in tqdm(range(pop_size)):
                x = np.reshape(cur_p_theta.rvs(), (self.dim, 1))
                controller = Controller(self.args, weights=x)
                fitness[j] = self.loss_func(controller)
                log_der = np.reshape(cur_grad(x), (len(theta), 1))
                grad_J += fitness * log_der / pop_size
                F += log_der @ log_der.T / pop_size

            theta += learn_rate * (np.linalg.inv(F) @ grad_J).flatten()
            reward[i] = self.get_reward_stats(theta[:self.dim], fitness)

        toc = time.perf_counter()
        print(f'Duration of training the controller: {toc - tic:0.4f} seconds')

        self.theta = theta
        return Controller(self.args).set_weight_array(self.get_mu(theta)), reward


    def grad_log_mvn_pdf(self, theta):
        '''
        theta is of the form: [mu1, mu2, sigma1 ^ 2, rho * sigma1 * sigma2, sigma2 ^ 2]
        '''
        Sigma_inv = np.linalg.inv(self.get_Sigma(theta))
        mu = self.get_mu(theta)
        mu_grad = lambda x: Sigma_inv @ (x - mu)
        Sigma_grad = lambda x: 0.5 * (Sigma_inv @ (x - mu) @ (x - mu).T @ Sigma_inv - Sigma_inv)
        return lambda x: np.concatenate([mu_grad(x).flatten(), Sigma_grad(x).flatten()[[0, 1, 3]]]) ## todo: 0,1,3???

    def get_mu(self, theta):
        return np.reshape(theta[:self.dim], (self.dim, 1))

    def get_Sigma(self, theta):
        Sigma = np.zeros(self.dim)
        for i in range(self.dim):
            for j in range(self.dim):
                Sigma[i,i] = theta
        return #2x2:  np.array([[1,2],[2,3]])
    # 3x3: np.array([[1,2,3],[2,4,5],[]])

