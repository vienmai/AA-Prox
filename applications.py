import numpy as np
import numpy.linalg as nla
from functions import *
from projectors import *



def quadratic(n, r, condidtion_number=1.0, noise=0.0, randseed=-1):
    if randseed > 0:
        np.random.seed(randseed)

    U = np.random.randn(n, r)
    U = np.linalg.qr(U)[0]
    V = np.random.randn(n, r)
    V = np.linalg.qr(V)[0]

    # eig = np.logspace(0.0, -2, num=r)
    eig = np.linspace(1.0, 1.0/np.sqrt(condidtion_number), num=r)
    S = np.diag(eig)
    A = U.dot(S.dot(V.T))

    A = A.dot(A.T)

    xopt = np.random.randn(n)

    b = -A.dot(xopt) + noise*np.random.randn(n)

    f = Quadratic(A, b)

    L2 = nla.norm(A, ord=2)

    x0 = 0.5*np.random.randn(n)
    fopt = f(xopt)

    return f, x0, L2, fopt

def nonnegative_least_squares(A, b, noise=0.01, lambdaL2=0, randseed=-1, normalizeA=False):
    """
    Generate a random instance of L1-regularized KL regression problem
            minimize_{x >= 0}  (1/2m)||Ax-b||^2 + 0.5*lambda * ||x||^2
    where
        A:  m by n real matrix
        b:  n vector of length m
        noise:  noise level to generate b = A * x + noise
        lambda: L2 regularization weight
        normalizeA: wether or not to normalize columns of A

    Return f, h, L, x0:
        f: f(x) = (1/2m)||Ax-b||^2 + 0.5*lamda * ||x||^2
        h: h(x) = (1/2)||x||_2^2
        L2 = (||AA^T||_2^2)/m
        x0: initial point, all-one vector
    """
    assert A.shape[0] == b.shape[0], "A and b size not matching"
    m, n = A.shape
    if normalizeA:
        A = A / A.sum(axis=0)   # scaling to make column sums equal to 1

    f = LeastSquares(A, b, lambdaL2)
    h = SquaredL2NormNonnegativeOrthan()

    L2norm = nla.norm(A, ord=2)
    L2 = L2norm**2/m

#    x0 = np.ones(n)
    x0 = np.zeros(n)
   
    return f, h, L2, x0


def logistic_regression_Linf(A, b, B=1., lambdaL2=0, normalizeA=False):
    """
    Example of logistic regression with Linf bounds
        minimize_x  f(x) = (1/m) * sum_{i=1}^m log(1 + exp(-b_i*(ai'*x))) + 0.5*lambda||x||_2^2
        subject to  x in R^n, and ||x||_inf <= B
    where
        A:  m by n data matrix
        b:  vector of length m
        lambda: L2 regularization weight

    # Return f, h, L, x0:
        f: f(x)
        h: h(x) = Shannon entropy (with L1 regularization as Psi)
        L2 = (||AA^T||_2^2)/m/4
        x0: initial point
    """
    assert A.shape[0] == b.shape[0], "A and b size not matching"
    m, n = A.shape
    if normalizeA:
        A = A / A.sum(axis=0)   # scaling to make column sums equal to 1

    f = LogisticRegression(A, b, lambdaL2)
    h = L2Linf(B)

    L2norm = nla.norm(A, ord=2)
    L2 = L2norm**2/m/4

    x0 = np.zeros(n)

    return f, h, L2, x0


def KL_nonneg_regL1(m, n, noise=0.01, lambdaL1=0, randseed=1, normalizeA=True):
    """
    Generate a random instance of L1-regularized KL regression problem
            minimize_{x >= 0}  D_KL(Ax, b) + lamda * ||x||_1
    where
        A:  m by n nonnegative matrix
        b:  nonnegative vector of length m
        noise:  noise level to generate b = A * x + noise
        lambda: L2 regularization weight
        normalizeA: wether or not to normalize columns of A

    Return f, h, L, x0:
        f: f(x) = D_KL(Ax, b)
        h: h(x) = Shannon entropy (with L1 regularization as Psi)
        L: L = max(sum(A, axis=0)), maximum column sum
        x0: initial point, scaled version of all-one vector
    """
    if randseed > 0:
        np.random.seed(randseed)
    A = np.random.rand(m, n)
    if normalizeA:
        A = A / A.sum(axis=0)   # scaling to make column sums equal to 1
    x = np.random.rand(n)
    b = np.dot(A, x) + noise * (np.random.rand(m) - 0.5)
    assert b.min() > 0, "need b > 0 for nonnegative regression."

    f = KLNonnegRegression(A, b)
    h = ShannonEntropyl1(lambdaL1)
    L = max(A.sum(axis=0))  # L = 1.0 if columns of A are normalized

    x0 = np.ones(n)
  
    return f, h, L, x0


