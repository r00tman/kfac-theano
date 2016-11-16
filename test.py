import numpy as np
import numpy.random as nprng
import scipy as sp
from scipy.linalg import solve
from scipy.special import expit

P = 100
W = nprng.randn(1, P)*1e-1

# W: 1xP, x: NxP -> Nx1
def getz(W, x):
    return expit(np.matmul(W, x.T).T)

# W: 1xP, z: Nx1, x: NxP -> NxP
def getJ(W, x):
    z = getz(W, x)  # Nx1
    J = x * np.tile(z*(1-z), (1, P))  # NxP
    return J

def getG(W, x):
    J = getJ(W, x)
    return np.matmul(J.T, J)/J.shape[0]  # PxP


def getUpdate(W, x, y, lambd):
    G = getG(W, x)  # PxP
    Gdamp = G + np.identity(P) * lambd

    J = getJ(W, x)  # NxP
    grad = np.mean(J * np.tile(y-getz(W, x), (1, P)), 0)

    upd = solve(Gdamp, grad)  # P
    return upd


def getLoss(W, x, y):
    return np.mean((getz(W, x)-y)**2)

N = 1000000

delta = 0.1

x1 = nprng.randn(N//2, P) - delta
y1 = np.ones((N//2,1))*0

x2 = nprng.randn(N//2, P) + delta
y2 = np.ones((N//2,1))*1

x = np.concatenate([x1, x2], 0)
y = np.concatenate([y1, y2], 0)

lambd = 0e-14

for it in range(100):
    print(it, getLoss(W, x, y), np.log10(np.mean(getJ(W, x)**2)))
    W += getUpdate(W, x, y, lambd)
