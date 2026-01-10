import numpy as np


class RandomProcess:
    def reset_states(self):
        pass

#Annealing the standard deviation term which results in the stochastic part of the OrnsteinUhlenbeckProcess
class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=0.2,
                 dt=1e-2, x0=None, size=1,
                 sigma_min=None, n_steps_annealing=1000):
        '''
        theta     : parameter which defines the mean reversion tendency
        mu        : the long-term mean
        sigma     : the standard deviation
        sigma_min : min value of standard deviation below which annealing is stopped
        x0        : initial value of state
        dt        : discrete timestep
        '''
        super(OrnsteinUhlenbeckProcess,
              self).__init__(mu=mu,
                             sigma=sigma,
                             sigma_min=sigma_min,
                             n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + \
            self.theta * (self.mu -
                          self.x_prev) * self.dt + (
                              self.current_sigma * np.sqrt(self.dt) *
                              np.random.normal(size=self.size)
                              )
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)