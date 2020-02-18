import numpy as np
import numpy.linalg as nla
import time


def AA_least_squares(R, Y, mk, reg):
    """
    Solve Anderson subproblem:
        minimize    ||(RR^T + reg I)alpha||^2
        subject to  sum(alpha) = 1
    where:
        R: mk by n residual matrix
        Y: mk by n matrix of iterates to be extrapolated
        mk: memory of AA
        reg: regularization term
        alpha: mk-vector of combination coefficients
    Return:
        Y^T alpha
    """
    if len(R) > mk + 1:
        R.pop(0)
    if len(Y) > mk + 1:
        Y.pop(0)

    R = np.asarray(R)
    RR = R.dot(R.T)
    normRR = nla.norm(RR, 2)
    RR = RR / normRR

    try:
        z = nla.solve(RR + reg * np.eye(R.shape[0]), np.ones(R.shape[0]))
    except nla.LinAlgError:
        z = nla.lstsq(RR + reg * np.eye(R.shape[0]), np.ones(R.shape[0]), -1)
        z = z[0]

    coeffs = z / sum(z)
    Y_matrix = np.asarray(Y[len(Y) - mk - 1:])
    return Y_matrix.T.dot(coeffs)


def AA_BPG(init, max_iters, depth, kernel, loss, step, norm_type=2, reg=1e-10,
           verbose=False, verbskip=1, max_runtime=10, stop_eps=0.):
    """
    To be added
    """
    if verbose:
        print("\n AA-BPG method\n")
        print("     k      F(x)")

    Fx = np.zeros(max_iters)
    Tx = np.zeros(max_iters)

    x = init.copy()
    Gx = []  # storing m fixed-point mapppings g()
    R = []  # Residual matrix storing g(x)-x
    fx, gradx = loss.func_grad(x)
    Fx[0] = fx + kernel.extra_Psi(x)

    if verbose:
        print("{0:6d}  {1:10.3e}".format(0, Fx[0]))

    y_ext = kernel.gradient(x)  # y0
    gx = y_ext - step * gradx  # g(y0)
    Gx.append(gx)
    R.append(gx - y_ext)  # first residual g(y0)-y0
    y_ext = gx  # y1`
    x = kernel.prox_map(-y_ext)  # x1 = projection(y1)
    fx, gradx = loss.func_grad(x)
    runtime = 0.0

    for idx in range(max_iters - 1):
        start = time.time()
        Fx[idx + 1] = fx + kernel.extra_Psi(x)
        if verbose and idx % verbskip == 0:
            print("{0:6d}  {1:10.3e}".format(idx + 1, Fx[idx + 1]))

        mk = min(depth, idx + 1)

        gx = kernel.gradient(x) - step * gradx
        Gx.append(gx)
        x_prox = kernel.prox_map(-gx)
        # grad_mapping = (x - x_prox) / step
        R.append(gx - y_ext)
        y_test = AA_least_squares(R, Gx, mk, reg)
        x_test = kernel.prox_map(-y_test)

        # assert x_test.min() >= 0., 'idx: %d \n  %.6f ' % (idx+1, x_test.min())
        f_test, grad_test = loss.func_grad(x_test)
        # if f_test - fx <= -0.5 * step * nla.norm(grad_mapping, norm_type)**2:
        if f_test - fx <= np.dot(gradx, x_prox - x) + kernel.divergence(x_prox, x)/step:
            y_ext = y_test
            x = x_test
            fx = f_test
            gradx = grad_test
        else:
            y_ext = gx
            x = x_prox
            fx, gradx = loss.func_grad(x)


        runtime += time.time() - start
        Tx[idx+1] = runtime

        if (runtime > max_runtime) or (abs(Fx[idx + 1] - Fx[idx]) < stop_eps):
            break

    Fx = Fx[0:idx + 1]
    Tx = Tx[0:idx+1]
    return x, Fx, Tx


def proximal_gradient(x0, max_iters, step, f, h, linesearch=False, max_step=1.,
                      ls_ratio=2, ls_adapt=True, verbose=False, verbskip=1, max_runtime=10, stop_eps=0.):

    if verbose:
        print("\nProximal gradient method with backtracking linesearch")
        print("     k      F(x)          Lk")

    Fx = np.zeros(max_iters)
    Tx = np.zeros(max_iters)

    x = x0
    runtime = 0.0

    for k in range(max_iters):
        start = time.time()
        fx, g = f.func_grad(x)
        Fx[k] = fx + h(x)

        if linesearch:
            x1 = h.prox_gardient(x, g, step)
            while f(x1) > fx + np.dot(g, x1 - x) + nla.norm(x1 - x)**2 / step:
                step = step / ls_ratio
                x1 = h.prox_gardient(x, g, step)
            x = x1

            if ls_adapt:
                step = max(step * ls_ratio, max_step)
        else:
            x = h.prox_gardient(x, g, step)

        runtime += time.time() - start
        Tx[k] = runtime

        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e} ".format(k, Fx[k]))

        # stopping criteria
        if runtime > max_runtime:
            break

        if k > 0 and abs(Fx[k] - Fx[k - 1]) < stop_eps:
            break

    Fx = Fx[0:k + 1]
    Tx = Tx[0:k + 1]
    return x, Fx, Tx


def gradient_descent(x0, max_iters, step, f, proj=lambda x: x, linesearch=False, max_step=1.,
                     ls_ratio=2, ls_adapt=True, verbose=False, verbskip=1, max_runtime=10, stop_eps=0.):

    if verbose:
        print("\nGradient descent method with backtracking linesearch")
        print("     k      F(x)     ")

    Fx = np.zeros(max_iters)
    Tx = np.zeros(max_iters)

    x = x0
    runtime = 0.0

    for k in range(max_iters):
        start = time.time()
        fx, g = f.func_grad(x)
        Fx[k] = fx

        if linesearch:
            x1 = proj(x - step * g)
            while f(x1) > fx + np.dot(g, x1 - x) + nla.norm(x1 - x)**2 / step:
                step = step / ls_ratio
                x1 = proj(x - step * g)
            x = x1

            if ls_adapt:
                step = max(step * ls_ratio, max_step)
        else:
            x = proj(x - step * g)

        runtime += time.time() - start
        Tx[k] = runtime

        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e} ".format(k, Fx[k]))

        # stopping criteria
        if runtime > max_runtime:
            break

        if k > 0 and abs(Fx[k] - Fx[k - 1]) < stop_eps:
            break

    Fx = Fx[0:k + 1]
    Tx = Tx[0:k + 1]
    return x, Fx, Tx


def accelerated_proximal_descent(init, max_iters, step, f, proximal=lambda x: x,
                                 strongcvx=0, opt_coeff=0, restart=False, max_runtime=10, stop_eps=0.):
    """
    Nesterov's accelerated gradient descent

    Parameters
    ----------
    grads: gradient oracle
    epsilon: tolerance for stopping condition
    init: where to start (otherwise random)

    Output
    ------
    xs: x values from each iteration
    """

    Tx = np.zeros(max_iters)
    Fx = np.zeros(max_iters)

    # initialization
    x_current = init

    y_current = x_current
    t_current = 1.0

    # history
    runtime = 0.0

    for k in range(max_iters):

        # history
        Fx[k] = f(x_current)

        start = time.time()
        # gradient update
        if strongcvx:
            x_next = proximal(y_current - step * f.gradient(y_current))
            y_next = x_next + opt_coeff * (x_next - x_current)
        else:
            t_next = .5 * (1 + np.sqrt(1 + 4 * (t_current**2)))
            x_next = proximal(y_current - step * f.gradient(y_current))
            y_next = x_next + (t_current - 1.0) / \
                (t_next) * (x_next - x_current)
            t_current = t_next

        runtime += time.time() - start
        Tx[k] = runtime

        if runtime > max_runtime:
            break

        # relative error stoping condition
        if nla.norm(x_next - x_current) <= stop_eps * nla.norm(x_current):
            break

        x_current = x_next
        y_current = y_next

        if restart:
            if k > 0 and Fx[k] > Fx[k-1]:
                # if np.dot(g, x - x_1) > 0:
                t_current = 1.0     # reset theta = 1 for updating with equality
                y_current = x_current
    Fx = Fx[0:k + 1]
    Tx = Tx[0:k + 1]
    return x_current, Fx, Tx
