from model import CamereVideo
import numpy as np
import random as rand
import time

class HillClimbing():
    def copie(self, x):
        return np.copy(x)
    def modifica(self, x, p=30):
        for _ in range(p):
            index = rand.choice(range(len(x)))
            x[index] = x[index] + rand.choice([-1,1])*self._pas
            if self._problema.valid(x): return x
        return None
    def valoare(self, x):
        x = x.reshape((3,2))
        return self._problema.F(x)
    def __init__(self, problema: CamereVideo, x=None, pas=0.5):
        self._problema = problema
        if x is None: self._x = self._problema.gen_nec(False).flatten()
        else: self._x = x.flatten()
        self._pas = pas
    def run_basic(self, p=30, timp_alocat=30):
        x = self.copie(self._x)
        timp = time.time()
        while True:
            r = self.modifica(self.copie(x), p)
            if r is None: break
            if self.valoare(r)>self.valoare(x):
                x=self.copie(r)
            delta_time = time.time()-timp
            if delta_time>timp_alocat: break
        return x
    def run_steep(self, p=30, n_iteratii=10, timp_alocat=30):
        x = self.copie(self._x)
        timp = time.time()
        while True:
            r = self.modifica(self.copie(x), p)
            if r is None: break
            for _ in range(n_iteratii):
                w = self.modifica(self.copie(r), p)
                if w is None: break
                if self.valoare(w)>self.valoare(r):
                    r=w
            if self.valoare(r)>self.valoare(x):
                x=r
            delta_time = time.time()-timp
            if delta_time>timp_alocat: break
        return x

    def _distributie_temporala(self, timp_alocat):
        T = []
        q = timp_alocat
        while q>0:
            t = rand.random()*timp_alocat/2
            q = q-t
            T.append(t)
        return np.array(T)
    def run_random(self, p=30, timp_alocat=30):
        T = self._distributie_temporala(timp_alocat)
        x = self.copie(self._x)
        x_optim = x
        timp1 = time.time()
        while True:
            t = rand.choice(T)
            timp2 = time.time()
            while True:
                r = self.modifica(self.copie(x), p)
                if r is None: break
                if self.valoare(r)>self.valoare(x):
                    x=r
                delta_time1 = time.time()-timp2
                delta_time2 = time.time()-timp1
                if delta_time1>t or delta_time2>timp_alocat: break
            if self.valoare(x)>self.valoare(x_optim):
                x_optim = x
            delta_time = time.time()-timp1
            if delta_time>timp_alocat: break
        return x_optim