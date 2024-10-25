import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

def construct(X, Y, ns, nt):
    Mr = np.zeros((ns, ns * nt))
    
    for i in range(ns):
        Mr[i : i + 1, i * nt : (i + 1) * nt] = np.ones((1, nt))
        
    Mc = np.identity(nt)
    for _ in range(ns - 1):
        Mc = np.hstack((Mc, np.identity(nt)))

    S = np.vstack((Mr, Mc))

    h = np.vstack((np.ones((ns, 1)) / ns, np.ones((nt, 1)) / nt))

    # Delete last row
    S = S[0 : (ns + nt - 1), :]
    h = h[0 : (ns + nt - 1), :]

    Omega = np.hstack((np.kron(np.identity(ns), np.ones((nt, 1))), np.kron(- np.ones((ns, 1)), np.identity(nt))))
    
    c_ = np.zeros((ns * nt, 1))

    for i in range(X.shape[1]):
        c_ += (Omega.dot(X[:, [i]])) * (Omega.dot(X[:, [i]]))

    cost = c_ + (Omega.dot(Y)) * (Omega.dot(Y)) 

    return S, h, Omega, c_, cost

def LinearProgram(cost, S, h):
    n = cost.shape[0]
    res = linprog(cost, A_ub = - np.identity(n), b_ub = np.zeros((n, 1)), A_eq = S, b_eq = h, method = 'simplex', options = {'maxiter': 20000})

    B = res.basis
    Bc = np.arange(n)

    for i in B:
        Bc = np.delete(Bc, np.where(Bc == i)[0][0])

    return B, Bc, res.x

def solve_Lasso(X, Y, Lambda):
    p = X.shape[1]

    lasso = Lasso(alpha = Lambda / X.shape[0], fit_intercept=False, tol = 1e-10, max_iter=100000000)
    lasso.fit(X, Y)

    beta_hat = lasso.coef_.reshape(p, 1)
    M = np.nonzero(beta_hat)[0]
    Mc = np.nonzero(beta_hat == 0)[0]
    sM = np.sign(beta_hat[M])
    return M, Mc, sM

def solve_ElasticNet(X, Y, Lambda, Gamma):
    n = X.shape[0]
    p = X.shape[1]

    elasticnet = ElasticNet(alpha=(Lambda+Gamma)/n, l1_ratio=Lambda/(Lambda+Gamma), fit_intercept=False, tol = 1e-10, max_iter=10000000)
    elasticnet.fit(X, Y)

    beta_hat = elasticnet.coef_.reshape(p, 1)
    M = np.nonzero(beta_hat)[0]
    Mc = np.nonzero(beta_hat == 0)[0]
    sM = np.sign(beta_hat[M])
    return M, Mc, sM

def checkKKT_Lasso(X, Y, M, beta_hat, Lambda):
    XM = X[:, M]
    sM = np.sign(beta_hat)[M, :]
    beta_hat_M = beta_hat[M, :]
    print('-----------------Check KKT Lasso-----------------')
    print(np.round((XM.T.dot(XM))).dot(beta_hat_M) - XM.T.dot(Y) + Lambda * sM)
    print('-----------------Check KKT Lasso-----------------')

def checkKKT_EN(X, Y, M, beta_hat, Lambda, Gamma):
    XM = X[:, M]
    sM = np.sign(beta_hat)[M, :]
    beta_hat_M = beta_hat[M, :]
    print('-----------------Check KKT Elastic Net-----------------')
    print(np.round((XM.T.dot(XM) + Gamma * np.identity(M.shape[0])).dot(beta_hat_M) - XM.T.dot(Y) + Lambda * sM))
    print('-----------------Check KKT Elastic Net-----------------')

def intersect(list_intervals_1, list_intervals_2):
    results = []
    
    if len(list_intervals_1) == 0 or len(list_intervals_2) == 0:
        return results

    for interval_1 in list_intervals_1:
        for interval_2 in list_intervals_2:
            if interval_1[1] < interval_2[0]:
                break
            if interval_1[0] > interval_2[1]:
                continue
            results.append([max(interval_1[0], interval_2[0]), min(interval_1[1], interval_2[1])])

    return results

def Lasso_after_DA(X, Y, ns, nt, Lambda):
    p = X.shape[1]

    H, h, cost = [construct(X, Y, ns, nt)[i] for i in [0, 1, 4]]
    
    T = LinearProgram(cost, H, h)[2]
    T = T.reshape(ns, nt)

    Omega = np.hstack((np.zeros((ns + nt, ns)), np.vstack((ns * T, np.identity(nt)))))

    X_tilde = Omega.dot(X)
    Y_tilde = Omega.dot(Y)
    
    return solve_Lasso(X_tilde, Y_tilde, Lambda)
    
def EN_after_DA(X, Y, ns, nt, Lambda, Gamma):
    p = X.shape[1]

    S, h, cost = [construct(X, Y, ns, nt)[i] for i in [0, 1, 4]]
    
    T = LinearProgram(cost, S, h)[2]
    T = T.reshape(ns, nt)

    Omega = np.hstack((np.zeros((ns + nt, ns)), np.vstack((ns * T, np.identity(nt)))))

    X_tilde = Omega.dot(X)
    Y_tilde = Omega.dot(Y)
    
    return solve_ElasticNet(X_tilde, Y_tilde, Lambda, Gamma)