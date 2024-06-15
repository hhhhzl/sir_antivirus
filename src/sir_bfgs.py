import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from matplotlib.lines import Line2D
import scipy.stats as stats
plt.rcParams['font.size'] = 12


def fun(pi, *args):
    time_steps = args[0]

    S = np.zeros(time_steps, dtype=np.float64)
    I = np.zeros(time_steps, dtype=np.float64)
    R = np.zeros(time_steps, dtype=np.float64)

    S[0] = 1 - args[1] - args[2]
    I[0] = args[1]
    R[0] = args[2]

    beta = args[3]
    sigma = args[4]
    gamma = args[5]

    delta_t = args[6]

    for i in range(time_steps - 1):
        dSdt = - beta * S[i] * I[i] - pi[i] * sigma * S[i] * R[i]
        dIdt = beta * S[i] * I[i] - (1 - pi[i]) * gamma * I[i] * R[i]
        dRdt = pi[i] * sigma * S[i] * R[i] + (1 - pi[i]) * gamma * I[i] * R[i]

        S[i + 1] = S[i] + dSdt * delta_t
        I[i + 1] = I[i] + dIdt * delta_t
        R[i + 1] = R[i] + dRdt * delta_t

    return np.sum(I)


class SIR:
    def __init__(
            self,
            initial_infected_prop,
            initial_recovered_prop,
            beta,
            sigma,
            gamma,
            time_steps,
            delta_t,
            **kwargs
    ):
        self.I0 = initial_infected_prop
        self.R0 = initial_recovered_prop
        self.S0 = 1 - self.I0 - self.R0

        self.delta_t = delta_t

        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma

        self.time_steps = time_steps

        # time steps to integrate over
        self.t = np.arange(0, time_steps, 1)

        # newly introduced parameters
        self.pi = np.full(time_steps, 0.5)

        self.S = np.zeros(self.time_steps)
        self.I = np.zeros(self.time_steps)
        self.R = np.zeros(self.time_steps)

        self.lambda_a = np.zeros(self.time_steps)
        self.lambda_b = np.zeros(self.time_steps)

        self.switch = np.zeros(self.time_steps)

    def run(self):

        self.S[0] = self.S0
        self.I[0] = self.I0
        self.R[0] = self.R0

        for i in range(self.time_steps - 1):
            dSdt = - self.beta * self.S[i] * self.I[i] - self.pi[i] * self.sigma * self.S[i] * self.R[i]
            dIdt = self.beta * self.S[i] * self.I[i] - (1 - self.pi[i]) * self.gamma * self.I[i] * self.R[i]
            dRdt = self.pi[i] * self.sigma * self.S[i] * self.R[i] + (1 - self.pi[i]) * self.gamma * self.I[i] * self.R[
                i]

            self.S[i + 1] = self.S[i] + dSdt * self.delta_t
            self.I[i + 1] = self.I[i] + dIdt * self.delta_t
            self.R[i + 1] = self.R[i] + dRdt * self.delta_t

        return self.S, self.I, self.R

    def lbd(self):

        for i in range(self.time_steps - 1, 0, -1):
            dlambda_adt = 0 - (self.lambda_a[i] * (
                        -self.beta * self.I[i] - self.pi[i] * self.sigma * (1 - self.I[i] - 2 * self.S[i])) +
                               self.lambda_b[i] * (self.beta * self.I[i] + (1 - self.pi[i]) * self.sigma * self.I[i]))
            dlambda_bdt = -1 - (self.lambda_a[i] * (-self.beta * self.S[i] + self.pi[i] * self.sigma * self.S[i]) +
                                self.lambda_b[i] * (self.beta * self.S[i] - (1 - self.pi[i]) * self.gamma * (
                                1 - self.S[i] - 2 * self.I[i])))

            self.lambda_a[i - 1] = self.lambda_a[i] - dlambda_adt * self.delta_t
            self.lambda_b[i - 1] = self.lambda_b[i] - dlambda_bdt * self.delta_t

        for i in range(self.time_steps - 1):
            self.switch[i] = self.lambda_b[i] * self.gamma * self.I[i] - self.lambda_a[i] * self.sigma * self.S[i]

        return self.lambda_a, self.lambda_b

    def plot_compare(self, show=None):
        self.run()
        self.lbd()

        pi = self.pi

        plt.figure()
        plt.plot(self.S, label=r'$S$', linestyle='-', color='blue')
        plt.plot(self.I, label=r'$I$', linestyle='-', color='orange')
        plt.plot(self.R, label=r'$R$', linestyle='-', color='green')
        plt.plot(self.pi, label=r'$\pi$', linestyle='-', color='red')
        self.pi = np.linspace(1, 0, self.time_steps)
        self.run()
        self.lbd()
        plt.plot(self.S, linestyle='--', color='blue')
        plt.plot(self.I, linestyle='--', color='orange')
        plt.plot(self.R, linestyle='--', color='green')
        plt.plot(self.pi, linestyle='--', color='red')
        plt.xlabel("Time (t)")
        plt.ylabel("Population Percentage")

        lg = plt.legend(loc='center right')
        plt.gca().add_artist(lg)

        line_styles = [Line2D([], [], linestyle='-', color='red'), Line2D([], [], linestyle='--', color='red')]
        line_labels = ['Optimal', 'Heuristic']

        plt.legend(line_styles, line_labels, loc='center left')

        plt.grid(True)
        plt.savefig('figures/state_compare.png', dpi=300)
        if show:
            plt.show()

        self.pi = pi
        self.run()
        self.lbd()

        plt.figure()
        plt.plot(self.switch, linestyle='-', color='brown')
        self.pi = np.linspace(1, 0, self.time_steps)
        self.run()
        self.lbd()
        plt.plot(self.switch, linestyle='--', color='brown')
        line_styles = [Line2D([], [], linestyle='-', color='brown'), Line2D([], [], linestyle='--', color='brown')]
        line_labels = ['Optimal', 'Heuristic']
        plt.legend(line_styles, line_labels, loc='center right')
        plt.xlabel("Time (t)")
        plt.ylabel(r"$\phi(t)$")
        plt.grid(True)
        plt.savefig('figures/switch_compare.png', dpi=300)
        if show:
            plt.show()

    def plot(self, show=False, **kwargs):
        b, s, g = round(kwargs['b'],2), round(kwargs['s'], 2), round(kwargs['g'],2)
        self.run()
        self.lbd()

        plt.figure()
        plt.plot(self.S, label=r'$S$')
        plt.plot(self.I, label=r'$I$')
        plt.plot(self.R, label=r'$R$')
        plt.plot(self.pi, label=r'$\pi$')
        plt.xlabel("Time (t)")
        plt.ylabel("Population Percentage")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'figures/state_{b}_{s}_{g}.png', dpi=300)
        if show:
            plt.show()

        plt.figure()
        plt.plot(self.lambda_a, label=r'$\lambda_{a}$')
        plt.plot(self.lambda_b, label=r'$\lambda_{b}$')
        plt.xlabel("Time (t)")
        plt.ylabel(r'$\lambda(t)$')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'figures/single/lambda_{b}_{s}_{g}.png', dpi=300)
        if show:
            plt.show()

        plt.figure()
        # plt.plot(np.zeros(self.time_steps))
        plt.plot(self.switch, label=r'$\phi(t)$')
        plt.xlabel("Time (t)")
        plt.ylabel(r"$\phi(t)$")
        plt.grid(True)
        plt.savefig(f'figures/single/switch_{b}_{s}_{g}.png', dpi=300)
        if show:
            plt.show()


if __name__ == '__main__':
    from benchmark.single_cluster_model import ARGS

    args = tuple(ARGS.values())
    T = ARGS.get('time_steps')
    res = minimize(fun, np.zeros(T), args, method=ARGS.get("method"), bounds=Bounds(lb=np.zeros(T), ub=np.ones(T)))
    model = SIR(**ARGS)
    model.pi = res['x']
    model.plot(show=False, b=ARGS.get('beta'), s=ARGS.get('sigma'), g=ARGS.get('gamma'))
    model.plot_compare()


    # gausian
    # lower, upper = 0, 1
    # mu, sigma = 0.5, 1
    # stata = stats.truncnorm(
    #     (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    # beta_l = stata.rvs(5)
    # sigma_l = stata.rvs(5)
    # gamma_l = stata.rvs(5)
    # count = 0
    # for beta in beta_l:
    #     for sigma in sigma_l:
    #         for gamma in gamma_l:
    #             # beta = 0.1
    #             # sigma = 0.1
    #             # gamma = 0.1
    #             delta_t = 1
    #             args = (T, I0, R0, beta, sigma, gamma, delta_t)
    #             bnds = Bounds(lb=np.zeros(T), ub=np.ones(T))
    #             res = minimize(fun, np.zeros(T), args, method='L-BFGS-B', bounds=bnds)
    #             # print("pi", res['x'])
    #             model = SIR(initial_infected_prop=I0, initial_recovered_prop=R0, beta=beta, sigma=sigma, gamma=gamma, time_steps=T, delta_t=delta_t)
    #             model.pi = res['x']
    #             model.plot(show=False, b=beta, s=sigma, g=gamma)
    #             print(f"Graph down: b:{beta}, s:{sigma}, g:{gamma} ==> {count}")
    #             count += 1

