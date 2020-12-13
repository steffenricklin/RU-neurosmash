import numpy as np
np.random.seed(2020)
from multiprocessing import Pool
from tqdm import tqdm
import time
from controller.Controller import *
from settings import *
from controller.ES_trainer import ES_abstract
from scipy import stats

class NES_trainer(ES_abstract):

    def __init__(self, loss_function, pop_size, learn_rate, args):
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
        sigma_values = np.triu(self.cov_matrix)[np.triu_indices(self.dim)]
        self.theta = np.append(self.weights, sigma_values)
        self.grad = self.grad_log_mvn_pdf
        self.learn_rate = learn_rate
        self.p_theta = lambda theta: stats.multivariate_normal(mean=self.get_mu(theta).flatten(), cov=self.get_Sigma(theta))


    def train(self, n_iter, parallel=False):
        '''
        Natural Evolution Strategy algorithm
        :param n_iter    (int) number of iterations
        :param parallel (bool) If true, run the Neurosmash environment in parallel

        :return trained controller (Controller) trained controller object with trained weights
        :return fitness  (matrix) fitness score for each member for each iteration
        '''
        pop_size, learn_rate = self.pop_size, self.learn_rate
        p_theta, theta = self.p_theta, self.theta
        grad = self.grad

        reward = np.zeros((n_iter,4))  # score of i) mean weight ii) best performer iii) worst performer iv) sampled population average
        print('using NES trainer for controller')
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
                controller = Controller(self.args)
                controller.set_weight_array(x)
                fitness[j] = self.loss_func(controller)
                log_der = np.reshape(cur_grad(x), (len(theta), 1))
                grad_J += fitness[j] * log_der / pop_size
                F += log_der.T @ log_der / pop_size

            theta += learn_rate * (np.linalg.inv(F) @ grad_J).flatten()
            reward[i] = self.get_reward_stats(theta[:self.dim], fitness)

        toc = time.perf_counter()
        print(f'Duration of training the controller: {toc - tic:0.4f} seconds')

        self.theta = theta
        self.weights = self.get_mu(theta)
        self.cov_matrix = self.get_Sigma(theta)
        trained_controller = Controller(self.args)
        trained_controller.set_weight_array(self.weights)
        return trained_controller, reward


    def grad_log_mvn_pdf(self, theta):
        '''
        theta is of the form: [mu1, mu2, sigma1 ^ 2, rho * sigma1 * sigma2, sigma2 ^ 2]
        '''
        Sigma_inv = np.linalg.inv(self.get_Sigma(theta))
        mu = self.get_mu(theta)
        mu_grad = lambda x: Sigma_inv @ (x - mu)
        Sigma_grad = lambda x: 0.5 * (Sigma_inv @ (x - mu) @ (x - mu).T @ Sigma_inv - Sigma_inv)
        idx_matrix = np.arange(1, (self.dim * self.dim) + 1).reshape((self.dim, self.dim))
        idx = np.triu(idx_matrix).flatten()
        idx = idx[idx > 0] - 1

        return lambda x: np.concatenate([mu_grad(x).flatten(), Sigma_grad(x).flatten()[idx]])

    def get_mu(self, theta):
        return np.reshape(theta[:self.dim], (self.dim, 1))

    def get_Sigma(self, theta):
        Sigma_values = theta[self.dim:]
        Sigma = np.zeros((self.dim,self.dim))
        ind_u = np.triu_indices(self.dim)
        Sigma[ind_u] = Sigma_values
        Sigma = Sigma + Sigma.transpose() - np.diag(np.diag(Sigma))
        return Sigma

