import random as rand
import numpy as np
import math
import time
from model import CamereVideo

def indentic(x, y):
    for e in x:
        if not e in y:
            return False
    return True

def exista(a, x):
    for e in a:
        if indentic(e, x):
            return True
    return False

class SimulatedAnnealing():
    def copie(self, x):
        return np.copy(x)
    def vecini(self, x):
        xt = np.array([x])
        x = self.copie(xt[0])
        for semn in [-1, 1]:
            for index in range(len(x)):
                x[index] += semn*self._pas
                if self._problema.valid(x):
                    xt = np.vstack((xt, x))
        return xt
    def q(self, x):
        x = x.reshape((3,2))
        return self._problema.F(x)
    def __init__(self, problema: CamereVideo, x=None, pas=0.5):
        self._problema = problema
        if x is None: self._x = self._problema.gen_nec(False).flatten()
        else: self._x = x.flatten()
        self._pas = pas
    def probabilitate(self, x, r):
        return math.exp((self.q(r)-self.q(x))/self.beta)
    def run(self, beta=0.7, z=15, timp_alocat=30, k=2):
        xt = np.array([self.copie(self._x)])
        t = 0
        x = xt[t]
        x_optim = x

        self.beta0 = rand.random()*rand.randint(10000, 1000000)
        self.beta = self.beta1 = beta*self.beta0
        timp = time.time()
        while True:
            for _ in range(z):
                r = rand.choice(self.vecini(self.copie(xt[t])))
                if self.q(r)>self.q(xt[t]):
                    xt = np.vstack((xt, r))
                if self.q(r)>self.q(x_optim):
                    x_optim = r
                if self.q(r)<=self.q(xt[t]):
                    u = rand.random()
                    if u <= self.probabilitate(xt[t], r):
                        xt = np.vstack((xt, r))
                        break
                    else:
                        xt = np.vstack((xt, xt[t]))
                
            self.beta = math.pow(self.beta1/self.beta0, k)*self.beta
            t, delta_time = t+1, time.time()-timp
            if self.beta<=1e-3 or delta_time>timp_alocat: break
        return x_optim