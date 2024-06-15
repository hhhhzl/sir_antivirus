import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from matplotlib.lines import Line2D
import pprint
import copy
import time
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from functools import partial
import logging
import numba
from joblib import Parallel, delayed
import os
from pathlib import Path
import pickle


plt.rcParams['font.size'] = 12
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class SIR_CLUSTER:
    ROOT_PATH = Path(__file__).resolve().parent.parent

    def __init__(self,
                 number_of_cluster=None,
                 S0_list=None,
                 I0_list=None,
                 beta_list=None,
                 sigma_list=None,
                 gamma_list=None,
                 tau_matrix=None,
                 phi_matrix=None,
                 time_step=None,
                 objective_weight=None,
                 parallel=False,
                 **kwargs
        ):

        # define number of clusters and initial S0, I0, R0 for each cluster
        self.number_of_cluster = number_of_cluster
        self.S0, self.I0, self.R0 = S0_list, I0_list, 1 - \
            np.add(S0_list, I0_list)
        self.control_space = 2 + self.number_of_cluster - 1

        # define parameters
        self.beta = beta_list
        self.sigma = sigma_list
        self.gamma = gamma_list
        self.tau = np.matrix(tau_matrix)
        self.weight = objective_weight
        self.phi = np.matrix(phi_matrix)

        # define time interval
        self.time_step = time_step
        self.t = np.arange(0, time_step, 1)

        # for optimization parameters
        self.mu = np.ones(
            (self.number_of_cluster, self.control_space, self.time_step))
        self.pi = self.transform(self.mu)
        for i in range(self.number_of_cluster):
            for k in range(self.time_step):
                self.pi[i, :, k] = [
                    1 / self.control_space for j in range(self.control_space)]

        # for space
        self.space = np.zeros((self.number_of_cluster * 3, time_step))
        self.space[:, 0] = [x for l in zip(
            self.S0, self.I0, self.R0) for x in l]

        # when t is large, implement multi-Threads
        if self.time_step >= 30:
            self.threads = True
        else:
            self.threads = False

        self.checked_inputs = False
        self.cluster_work_root = None

        # define plots
        self.figs, self.axs, self.lines = [], [], []

    # ---------------------------Main algorithms----------------------------#
    def transform(self, mu):

        pi = np.ones(
            (self.number_of_cluster, self.control_space, self.time_step))

        # parallelize
        for i in range(self.number_of_cluster):
            for t in range(self.time_step):
                numerator = np.exp(mu[i, :, t])
                denominator = np.sum(numerator)
                pi[i, :, t] = numerator / denominator
        return pi

    def n_by_n(self, pi):

        upper = np.hstack((np.zeros((pi.shape[0], 1)), np.triu(pi, k=0)))
        lower = np.hstack((np.tril(pi, k=-1), np.zeros((pi.shape[0], 1))))
        arr = np.add(upper, lower)

        return arr

    def run(self, mu):
        pi = self.transform(mu)
        space = copy.deepcopy(self.space)
        for t in range(len(self.t) - 1):
            delta = []
            # parallelize
            for i in range(self.number_of_cluster):
                dot = [
                    - self.beta[i] * space[i * 3, t] * space[i * 3 + 1, t] - pi[i, 0, t] * self.sigma[i] * space[
                        i * 3 + 2, t] * space[i * 3, t],
                    self.beta[i] * space[i * 3, t] * space[i * 3 + 1, t] - pi[i, 1, t] * self.gamma[i] * space[
                        i * 3 + 2, t] * space[i * 3 + 1, t]
                ]
                sum = [0, 0]
                n_by_n = self.n_by_n(pi[:, 2:, t])
                for j in range(self.number_of_cluster):
                    if i == j:
                        pass
                    else:
                        term = [n_by_n[j, i] * self.tau[j, i] * space[i * 3, t] * space[j * 3 + 2, t],
                                n_by_n[j, i] * self.phi[j, i] * space[i * 3 + 1, t] * space[j * 3 + 2, t]]
                        sum = np.add(sum, term)
                dot = np.subtract(dot, sum)
                dot = np.append(dot, -np.sum(dot))
                delta = np.append(delta, dot)
            # synchronization()

            space[:, t + 1] = np.add(space[:, t], delta)

        return space

    def loss(self, space):

        object_t = space[1::3, :]
        integral = np.sum(object_t, axis=1)
        loss = np.dot(self.weight, integral)

        return loss

    def fun(self, input, *args):

        mu = input.reshape(self.number_of_cluster,
                           self.number_of_cluster + 1, self.time_step)
        space = self.run(mu)
        loss = self.loss(space)

        return loss

    def plot(self, mu):

        self.figs, self.axs, self.lines = [], [], []

        for i in range(self.number_of_cluster):
            fig, ax = plt.subplots(sharex=True, sharey=True)
            self.figs.append(fig)
            self.axs.append(ax)

        space = self.run(mu)

        self.pi = self.transform(mu)

        for i in range(self.number_of_cluster):

            legend = [r"$S$", r"$I$", r"$R$", r"$\omega$", r"$\epsilon$"] + \
                [r"$p_{{{}{}}}$".format(j+1, i+1)
                 for j in range(self.number_of_cluster) if j+1 != i+1]
            line = [self.axs[i].plot(self.t[:-1], self.pi[i, 0, :-1], label=legend[j])[0]
                    for j in range(3 + self.control_space)]
            self.axs[i].set_ylim(-0.1, 1.1)
            self.axs[i].set_xlim(0, self.time_step)
            self.axs[i].legend(loc='upper right')
            self.axs[i].set_xlabel("Time (t)")
            self.axs[i].set_ylabel("Population Percentage")
            self.lines.append(line)

            for j in range(3 + self.control_space):
                if j < 3:
                    self.lines[i][j].set_ydata(space[i * 3 + j, :-1])
                    line += [self.lines[i][j]]
                else:
                    self.lines[i][j].set_ydata(self.pi[i, j - 3, :-1])
                    line += [self.lines[i][j]]

            self.axs[i].relim()
            self.axs[i].autoscale_view(True, True, True)
            self.axs[i].set_title('Cluster {}'.format(i + 1))
            self.figs[i].savefig('./figures_r/cluster/s2_7_cluster {}'.format(i + 1), dpi=300)
        plt.show()

        return


if __name__ == "__main__":
    from benchmark.heterogeneous_clusters_model import ARGS1, ARGS2

    # Scenario I: different beta, everything else stay the same
    T, n_cluster, method, lr = ARGS1.get("T"), ARGS1.get("number_of_cluster"), ARGS1.get('method'), ARGS1.get('lr')
    net = SIR_CLUSTER(**ARGS1)
    res = minimize(net.fun, np.ones(n_cluster * (n_cluster + 1) * T), args=[], method=method, tol=lr)
    mu = res['x'].reshape(net.number_of_cluster, net.number_of_cluster + 1, net.time_step)
    net.plot(mu)

    # Scenario II: different weights, everything else stay the same
    # note: no single bang when beta_list=[0.1, 0.15, 0.2]
    net = SIR_CLUSTER(ARGS2)
    T, n_cluster, method, lr = ARGS2.get("T"), ARGS2.get("number_of_cluster"), ARGS2.get('method'), ARGS2.get('lr')
    res = minimize(net.fun, np.ones(n_cluster * (n_cluster + 1) * T), args=[], method=method, tol=lr)
    mu = res['x'].reshape(net.number_of_cluster,net.number_of_cluster + 1, net.time_step)
    net.plot(mu)
