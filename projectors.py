import numpy as np


def nonneg_projection(x):
    """
    Proximal operator for enforcing non-negativity (indicator function over the set x >= 0)
    Parameters
    ----------
    x : array_like
        The starting or initial point used in the proximal update step
    Returns
    -------
    theta : array_like
        The parameter vector found after running the proximal update step
    """

    return np.maximum(x, 0.)


def box_projection(x, lower=-1., upper=1.):
    """
    Proximal operator for enforcing box constraint (lower =< x <= upper)
    Parameters
    ----------
    x : array_like
        The starting or initial point used in the proximal update step
    lower: lower bound
    upper: upper bound

    Returns
    -------
    theta : array_like
        The parameter vector found after running the proximal update step
    """
    np.clip(x, lower, upper, out=x)
    return x


def l1_ball_projection(v):
    u = np.abs(v)
    if u.sum() <= 1.0:
        return v
    w = l2_simplex_projection(u)
    w *= np.sign(v)
    return w


def l2_simplex_projection(s):
    """
    Projection onto the unit simplex
        C = {x_i >=0 : sum(x_i) =1}
    # Code taken from https://gist.github.com/daien/1272551
    """
    if np.sum(s) <= 1 and np.alltrue(s >= 0):
        return s

    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(s)[::-1]  # sort and reverse
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - 1) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    return np.maximum(s-theta, 0.)



