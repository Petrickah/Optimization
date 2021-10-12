import pandas as pd
import numpy as np
import random as rand

# Ecuatia unei drepte
def ecuatiedr(p1, p2):
    p = np.array([p1[0], p1[1], p2[0], p2[1]]).reshape((2, 2))
    return np.array([
        np.diff(p.transpose()[1]),
        -np.diff(p.transpose()[0]),
        -(p[1][0]*np.diff(p.transpose()[1])-p[1][1]*np.diff(p.transpose()[0]))
    ])

# Intersectia a doua laturi
def intersect(lat1: np.ndarray, lat2: np.ndarray):
    # A * x + B = 0
    A = np.vstack((lat1, lat2))[:,:2]
    B = np.vstack((lat1, lat2))[:,-1]
    detA = np.linalg.det(A)                 # |A| - determinantul lui A
    p = np.array([np.Inf, np.Inf])          # P - intersectia dreptelor
    if detA!=0:                             # |A|!=0
        invA = np.linalg.inv(A)             # A^(-1) - inversa lui A
        p = np.dot(invA, B)
        p = np.array([p[0], p[1]])
    return p

class CamereVideo():
    def _ecuatii(self):
        laturi = np.ndarray((3,1), dtype=np.int64)
        for li in np.array(self._date):
            l = np.array([
                [li[0],li[1]],
                [li[2],li[3]]
            ])
            l = np.array([
                np.diff(l.transpose()[1]), # yA-yB
                -np.diff(l.transpose()[0]), # -(xA-xB)
                -(l[1][0]*np.diff(l.transpose()[1])-l[1][1]*np.diff(l.transpose()[0])) # -(xB(yA-yB)-yB(xA-xB))
            ])
            laturi = np.hstack((laturi, l))
        return laturi[:,1:].transpose() # tablou cu n linii, fiecare linie este o ecuatie a unei drepte
    def __init__(self, date: str):
        self._date = pd.read_csv(date, index_col=0)
        self._date_intrare = pd.DataFrame(
            np.hstack(
                (self._ecuatii(), np.array(self._date))
            ), 
            index=self._date.index, 
            columns=["a", "b", "c", "x1", "y1", "x2", "y2"]
        )
        self._intersectii = self.__intersectii()
        self._p = np.vstack((np.unique(self._intersectii[0]), np.unique(self._intersectii[1])))
        self._dr = None
    # Date intrare
    def get_date(self):
        return self._date_intrare
    # Necunoscuta
    def necunoscuta(self, cam1: np.array, cam2: np.array, cam3: np.array):
        return np.hstack((cam1, cam2, cam3)).transpose()
    def camera(self, cam: tuple):
        return np.array([[cam[0], cam[1]]]).transpose()
    def __intersectii(self):
        _intersectii = []
        ecuatii = np.array(self._date_intrare.iloc[:,:3])
        for l1 in ecuatii:
            for l2 in ecuatii[1:]:
                p = -intersect(l1, l2)
                if not (p[0]==-np.inf and p[1]==-np.inf):
                    p[0] = -p[0] if p[0]<=0 else p[0]
                    p[1] = -p[1] if p[1]<=0 else p[1]
                    p = [p[0], p[1]]
                    if not p in _intersectii:
                        _intersectii.append(p)
        _intersectii = np.array(_intersectii).transpose()
        return _intersectii
    def get_intersectii(self):
        return self._intersectii
    # Restrictii necunoscuta
    def valid(self, nec:np.array, verbose=False):
        self._intersectii = self.get_intersectii()
        ok = True
        n_camere = nec.shape[0]//2
        nec = nec.reshape((n_camere, 2))
        for cam in nec:
            if verbose: print(cam, ':', [self._p[0].max(), cam[1]], end=' -> ')
            # d = ecuatiedr(np.array([self._p[0].max(), cam[1]]), cam).flatten()
            n=0
            for lat in self._date_intrare.values:
                x = np.sort(lat[3:].reshape((2,2)).transpose()[0])
                y = np.sort(lat[3:].reshape((2,2)).transpose()[1])
                if y[0]<cam[1] and cam[1]<y[1] and np.diff(x)==0 and x[0]>cam[0]:
                    n+=1
            ok = ok and n%2==1
            if verbose: print(n, n%2==1)
        return ok
    # Necunoscuta aleatorie
    def gen_nec(self, verbose=True):
        k=0
        while True:
            if verbose: print("Pasul: ", k+1)
            camere=[]
            for i in range(3):
                l = rand.choice(list(self._date.index))
                latura = np.array(self._date.loc[l,:]).reshape((2, 2)).transpose()
                u1, u2 = rand.random(), rand.random()
                x, y = np.sort(latura[0]), np.sort(latura[1])
                minim, maxim = np.array([
                    np.array(self._date.min())[0], 
                    np.array(self._date.min())[1]
                ]), np.array([
                    np.array(self._date.max())[0], 
                    np.array(self._date.max())[1]
                ]) 
                if np.diff(x)==0: c = self.camera((minim[0]+u1*maxim[0], y.min()+u2*y.max()))
                else: c = self.camera((x.min()+u1*y.max(), minim[1]+u2*maxim[1]))
                camere.append(c+rand.random())
            x = np.array(camere)
            x = self.necunoscuta(x[0], x[1], x[2])
            if self.valid(x.flatten(), verbose=verbose): break
            k=k+1
        return x
    def __dreptunghiuri(self, x):
        self._dr = []
        for cam in x:
            diag, ok = [], True
            for axis, a in enumerate(cam):
                index, intv = 0, []
                while index<len(self._p[axis])-1:
                    if self._p[axis,index]<a and a<self._p[axis,index+1]:
                        intv = [self._p[axis,index], self._p[axis,index+1]]
                    index+=1
                diag.append(intv) 
            for d in self._dr:
                indentic = True
                for index, k in enumerate(np.array(diag).transpose().flatten()):
                    if k!=d[index]: indentic = False
                if indentic: ok = False
            if ok: self._dr.append(np.array(diag).transpose().flatten())
        return np.array(self._dr).reshape((-1,2,2))
    # Functie obiectiv
    def F(self, x):
        # Dreptunghiuri
        self._dr = self.__dreptunghiuri(x)
        # Arie dreptunghiuri
        self._arie=0
        for d in self._dr:
            self._arie += np.abs(np.diff(d.transpose()[0]))*np.abs(np.diff(d.transpose()[1]))
        return self._arie[0]
    # Diagrama incapere
    def plot(self, x):
        from plotly import graph_objs as go

        fig = go.Figure()
        [
            fig.add_trace(go.Scatter(
                x=np.array(self._date_intrare.loc[lat,'x1':]).reshape((2,2)).transpose()[0],
                y=np.array(self._date_intrare.loc[lat,'x1':]).reshape((2,2)).transpose()[1],
                mode="lines", line=dict(color="brown"), name="Latura Incapere %i" % (index+1)
            ))
            for index, lat in enumerate(self._date_intrare.index)
        ]
        [fig.add_trace(go.Scatter(x=[camera[0]],y=[camera[1]],mode="markers", name="Camera video %i" % (index+1))) for index, camera in enumerate(x)]
        fig.add_trace(go.Scatter(
            x=self._intersectii[0],
            y=self._intersectii[1],
            mode="markers",
            marker=dict(color="blue"),
            name='Intersectii'
        ))
        fig.add_trace(go.Scatter(
            x=self._p[0],
            y=self._p[1],
            mode="lines",
            line=dict(color="red"),
            name="Intervale incapere"
        ))
        [
            fig.add_trace(go.Scatter(
                x=d.transpose()[0],
                y=d.transpose()[1],
                mode="lines",
                name="Diagonala %i" % (index+1)
            ))
            for index, d in enumerate(self._dr)
        ]
        fig.show()