import numpy as np

def gen_data(n, m, p, true_beta_s, true_beta_t):
    Xs = np.random.normal(loc = 0, scale = 1, size = (n, p))
    Xt = np.random.normal(loc = 0, scale = 1, size = (m, p))

    mu_s = Xs.dot(true_beta_s)
    mu_t = Xt.dot(true_beta_t)

    Ys = mu_s + np.random.normal(loc = 0, scale = 1, size = (n, 1))
    Yt = mu_t + np.random.normal(loc = 0, scale = 1, size = (m, 1))

    Sigma_s = np.identity(n)
    Sigma_t = np.identity(m)

    return Xs, Xt, Ys, Yt, Sigma_s, Sigma_t