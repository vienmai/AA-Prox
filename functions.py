import numpy as np
import numpy.linalg as nla


class Quadratic:
    """
    f(x) = (1/2) x'Ax + b'x
    """

    def __init__(self, A, b):
        assert len(b) == A.shape[0], "Quadratic: len(b) != m"
        self.n = A.shape[1]
        self.A = A
        self.b = b

    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def gradient(self, x):
        return self.func_grad(x, flag=1)

    def func_grad(self, x, flag=2):
        assert x.size == self.n, "Quadratic: x.size not equal to n."
        Ax = np.dot(self.A, x)

        if flag == 0:
            fx = 0.5 * x.dot(Ax) + self.b.dot(x)
            return fx
        g = Ax + self.b

        if flag == 1:
            return g

        # return both function value and gradient
        fx = 0.5 * x.dot(Ax) + self.b.dot(x)
        return fx, g

    def inv_grad_map(self, y):
        return np.linalg.solve(self.A, y - self.b)

    def line_search(self, grad, d):
        return -np.dot(grad, d)/np.dot(d, self.A.dot(d))


class MatrixLeastSquares:
    """
    f(X) = (1/2) || A - XX^T||_F^2
    where:
        A: n by n symmetric matrix
        X: n by r low rank matrix
    """

    def __init__(self, A):
        assert A.shape[0] == A.shape[1]
        self.A = A
        self.n = A.shape[0]
        self.normAsq = nla.norm(A, 'fro')**2

    def __call__(self, X):
        return self.func_grad(X, flag=0)

    def gradient(self, X):
        return self.func_grad(X, flag=1)

    def func_grad(self, X, flag=2):
        assert X.shape[
            0] == self.n, "Nonnegative matrix factorization: numrows(X) not equal to n."
        XtX = np.dot(X.T, X)
        AX = np.dot(self.A, X)
        if flag == 0:
            fx = 0.5 * (self.normAsq + nla.norm(XtX, 'fro') ** 2 - 2 * np.trace(np.dot(AX.T, X)))
            return fx

        g = 2*(np.dot(X, XtX) - AX)
        if flag == 1:
            return g

        # return both function value and gradient
        fx = 0.5 * (self.normAsq + nla.norm(XtX, 'fro') ** 2 - 2 * np.trace(np.dot(AX.T, X)))
        return fx, g


class KLNonnegRegression:
    """
    f(x) = D_KL(Ax, b) for linear inverse problem A * x = b
    """

    def __init__(self, A, b):
        assert A.shape[0] == b.shape[0], "A and b size not matching"
        self.A = A
        self.b = b
        self.m = A.shape[0]
        self.n = A.shape[1]

    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def gradient(self, x):
        return self.func_grad(x, flag=1)

    def func_grad(self, x, flag=2):
        assert x.size == self.n, "NonnegRegression: x.size not equal to n."
        Ax = np.dot(self.A, x)
        if flag == 0:
            fx = sum(Ax * np.log(Ax / self.b) - Ax + self.b)
            return fx

        # use array broadcasting
        g = (np.log(Ax / self.b).reshape(self.m, 1) * self.A).sum(axis=0)
        # same as the following code
        # g = np.zeros(x.shape)
        # for i in range(self.m):
        #    g += np.log(Ax[i]/self.b[i]) * self.A[i,:]
        if flag == 1:
            return g

        # return both function value and gradient
        fx = sum(Ax * np.log(Ax / self.b) - Ax + self.b)
        return fx, g


#######################################################################


class LeastSquares:
    """
    f(x) = 1/2/m ||Ax-b||^2 + 0.5*alpha||x||^2
    """

    def __init__(self, A, b, alpha=0):
        assert len(b) == A.shape[0], "Least squares: len(b) != m"
        assert alpha >= 0, "Least squares: regularizer must be nonnegative"
        self.m = A.shape[0]
        self.n = A.shape[1]
        self.reg = alpha
        self.A = A
        self.b = b

    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def gradient(self, x):
        return self.func_grad(x, flag=1)

    def func_grad(self, x, flag=2):
        assert x.size == self.n, "Least squares: x.size not equal to n."
        Ax_b = np.dot(self.A, x) - self.b

        if flag == 0:
            fx = 0.5 * nla.norm(Ax_b)**2 / self.m + 0.5 * self.reg * x.dot(x)
            return fx
        g = self.A.T.dot(Ax_b) / self.m + self.reg * x

        if flag == 1:
            return g

        # return both function value and gradient
        fx = 0.5 * nla.norm(Ax_b)**2 / self.m + 0.5 * self.reg * x.dot(x)
        return fx, g

##########################################################################


# class logistic_regression:
#     def __init__(self, A, b, alpha):
#         assert len(b) == A.shape[0], "Logistic regression: len(b) != m"
#         assert alpha >= 0., "Logistic regression: regularizer must be nonnegative"
#         self.m = A.shape[0]
#         self.n = A.shape[1]
#         self.reg = alpha
#         self.A = A
#         self.b = b
#
#     def phi(self, t):
#         # logistic function, returns 1 / (1 + exp(-t))
#         idx = t > 0
#         out = np.empty(t.size, dtype=np.float)
#         out[idx] = 1./(1. + np.exp(-t[idx]))
#         exp_t = np.exp(t[~idx])
#         out[~idx] = exp_t/(1. + exp_t)
#         return out
#
#     def func(self, x):
#         assert x.size == self.n, "Logistic regression: x.size not equal to n"
#         z = self.A.dot(x)
#         yz = self.b*z
#         idx = yz > 0
#         out = np.zeros_like(yz)
#         out[idx] = np.log(1. + np.exp(-yz[idx]))
#         out[~idx] = (-yz[~idx] + np.log(1 + np.exp(yz[~idx])))
#         out = out.sum()/self.m + 0.5*self.reg*x.dot(x)
#         return out

    # def grad(self, x):
    #     assert x.size == self.n, "Logistic regression: x.size not equal to n"
    #     z = self.A.dot(x)
    #     z = self.phi(self.b*z)
    #     z0 = (z - 1.)*self.b
    #     return self.A.T.dot(z0)/self.m + self.reg*x


#########################################################################


class LogisticRegression:
    """
    f(x) = (1/m)*sum_{i=1}^m log(1 + exp(-b_i*(ai'*x))) + 0.5*alpha*||x||^2 with ai in R^n, bi in R
    """

    def __init__(self, A, b, alpha):
        assert len(b) == A.shape[0], "Logistic Regression: len(b) != m"
        assert alpha >= 0., "Logistic Regression: regularizer must nonnegative"
        self.bA = np.reshape(b, [len(b), 1]) * A
        self.m = A.shape[0]
        self.n = A.shape[1]
        self.reg = alpha

    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def gradient(self, x):
        return self.func_grad(x, flag=1)

    def func_grad(self, x, flag=2):
        assert x.size == self.n, "Logistic Regression: x.size not equal to n"

        bAx = np.dot(self.bA, x)

        loss = - bAx
        mask = bAx > -50
        loss[mask] = np.log(1. + np.exp(-bAx[mask]))
        f = np.sum(loss) / self.m + 0.5 * self.reg * x.dot(x)

        if flag == 0:
            return f

        p = -1. / (1. + np.exp(bAx))
        g = np.dot(p, self.bA) / self.m + self.reg * x

        if flag == 1:
            return g

        return f, g


class LegendreFunction:
    """
    Function of Legendre type, used as the kernel of Bregman divergence.
    Include an extra Psi(x) for convenience of composite optimization.
    """

    def __call__(self, x):
        assert 0, "LegendreFunction: __call__(x) is not defined."

    def extra_Psi(self, x):
        return 0

    def gradient(self, x):
        assert 0, "LegendreFunction: gradient(x) is not defined."

    def divergence(self, x, y):
        """
        Return D(x,y) = h(x) - h(y) - <h'(y), x-y>
        """
        assert 0, "LegendreFunction: divergence(x,y) is not defined."

    def prox_map(self, g):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + h(x) }
        """
        assert 0, "LegendreFunction: prox_map(x, L) is not defined."

    def div_prox_map(self, y, g, step):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> +  D(x,y)/step  }
        """
        assert 0, "LegendreFunction: div_prox_map(y, g, step) is not defined."



class ShannonEntropy(LegendreFunction):
    """
    h(x) = sum_{i=1}^n x[i]*log(x[i]) for x >= 0, note h(0) = 0
    """

    def __init__(self, delta=1e-20):
        self.delta = delta

    def __call__(self, x):
        assert x.min() >= 0, "ShannonEntropy takes nonnegative arguments."
        xx = np.maximum(x, self.delta)
        return sum(xx * np.log(xx))

    def gradient(self, x):
        assert x.min() >= 0, "ShannonEntropy takes nonnegative arguments."
        xx = np.maximum(x, self.delta)
        return 1.0 + np.log(xx)

    def divergence(self, x, y):
        assert x.shape == y.shape, "Vectors x and y are of different shapes."
        assert x.min() >= 0 and y.min() >= 0, "Some entries are negative."
        # for i in range(x.size):
        #    if x[i] > 0 and y[i] == 0:
        #        return np.inf
        return sum(x * np.log((x + self.delta) / (y + self.delta))) + (sum(y) - sum(x))

    def prox_map(self, g):
        """
        Return grad(h^*) = argmin_{x >= 0} { <g, x> + h(x) }
        """
        return np.exp(-g - 1)

    def div_prox_map(self, y, g, step):
        """
        Return argmin_{x >= 0} { <g, x> +  D(x,y)/step }
        Beck eq. (9.5) or p_lambda(x) in DescentLemma eq.(15)
        """
        assert y.shape == g.shape, "Vectors y and g are of different sizes."
        assert y.min() >= 0 and step > 0, "Some entries of y are negavie."
        # gg = g/L - self.gradient(y)
        # return self.prox_map(gg)
        return y * np.exp(-step * g)


class ShannonEntropySimplex(ShannonEntropy):
    """
    h(x) = sum_{i=1}^n x[i]*log(x[i]) for x >= 0, note h(0) = 0
    used in the context of  min_{x in C } f(x) where C is standard simplex
    """

    def prox_map(self, g):
        """
        Return argmin_{x in C} { <g, x> + h(x) } where C is unit simplex
        Beck eq(9.6)
        """
        x = np.exp(-g - 1)
        return x / sum(x)

    def div_prox_map(self, y, g, step):
        """
        Return argmin_{x in C} { <g, x> + d(x,y)/step } where C is unit simplex
        Beck eq. (9.5) or DescentLemma eq(14) f = Indicator(C)
        """
        assert y.shape == g.shape, "Vectors y and g are of different shapes."
        assert y.min() > 0 and step > 0, "prox_map needs positive arguments."
        x = y * np.exp(-step * g)
        return x / sum(x)


class ShannonEntropyl1(ShannonEntropy):
    """
    h(x) = sum_{i=1}^n x[i]*log(x[i]) for x >= 0, note h(0) = 0
    used in the context of  min_{x >=0 } f(x) + lamda * ||x||_1
    """

    def __init__(self, lamda=0, delta=1e-20):
        ShannonEntropy.__init__(self, delta)
        self.lamda = lamda

    def extra_Psi(self, x):
        """
        return lamda * ||x||_1
        """
        return self.lamda * x.sum()

    def prox_map(self, g):
        """
        Return argmin_{x >= 0} { lamda * ||x||_1 + <g, x> + h(x) }
        """
        return ShannonEntropy.prox_map(self, self.lamda + g)

    def div_prox_map(self, y, g, step):
        """
        Return argmin_{x >= 0} { lamda * ||x||_1 + <g, x> + D(x,y)/step }
        """
        return ShannonEntropy.div_prox_map(self, y, self.lamda + g, step)

class SquaredL2Norm(LegendreFunction):
    """
    h(x) = (1/2)||x||_2^2
    """

    def __call__(self, x):
        return 0.5 * np.dot(x, x)

    def gradient(self, x):
        return x

    def divergence(self, x, y):
        assert x.shape == y.shape, "SquaredL2Norm: x and y not same shape."
        xy = x - y
        return 0.5 * np.dot(xy, xy)

    def prox_map(self, g):
        return -g

    def div_prox_map(self, y, g, step):
        assert y.shape == g.shape and step > 0, "Vectors y and g not same shape."
        return y - step * g


class SquaredL2NormNonnegativeOrthan(SquaredL2Norm):
    """
    h(x) = (1/2)||x||_2^2 where x >= 0.
    """

    def prox_map(self, g):
        return np.maximum(-g, 0.)

    def div_prox_map(self, y, g, step):
        assert y.shape == g.shape and step > 0, "Vectors y and g not same shape."
        return self.prox_map(step * g - y)


class L2Linf(LegendreFunction):
    """
    usng h(x) = (1/2)||x||_2^2 in solving problems of the form

        minimize    f(x)
        subject to  ||x||_inf <= B

    """

    def __init__(self, B=1.):
        self.B = B

    def __call__(self, x):
        return 0.5 * np.dot(x, x)

    def gradient(self, x):
        """
        gradient of h(x) = (1/2)||x||_2^2
        """
        return x

    def divergence(self, x, y):
        """
        Bregman divergence D(x, y) = (1/2)||x-y||_2^2
        """
        assert x.shape == y.shape, "L2L1Linf: x and y not same shape."
        xy = x - y
        return 0.5 * np.dot(xy, xy)

    def prox_map(self, g):
        """
        Return argmin_{x in C} { <g, x> + h(x) }
        """
        x = -g
        np.clip(x, -self.B, self.B, out=x)
        return x

    def div_prox_map(self, y, g, step):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + D(x,y)/step  }
        """
        assert y.shape == g.shape and step > 0, "Vectors y and g not same shape."
        return self.prox_map(step * g - y)


