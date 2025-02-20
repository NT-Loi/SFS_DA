import numpy as np
import sfs_da.util as util
import sfs_da.pivot as pivot 
import sfs_da.conditioning as conditioning

def divide_and_conquer_Lasso(X, a, b, ns, nt, Lambda, zmin, zmax):
    list_intervals = []
    list_M = []

    zuv = zmin

    while zuv <= zmax:
        Y_zuv = a + b * zuv

        # Solve Linear Program to obtain Transportation Map for OT
        H, h, Theta, c_, cost = util.construct(X, Y_zuv, ns, nt)
        Bu, Bcu, Tu = util.LinearProgram(cost, H, h)
        Tu = Tu.reshape(ns, nt)
        Omega_u = np.hstack((np.zeros((ns + nt, ns)), np.vstack((ns * Tu, np.identity(nt)))))

        interval_u = conditioning.conditioning_Zu(a, b, c_, Theta, Bu, Bcu, H)

        # Select the interval containing the data point that we are currently considering.
        for i in interval_u:
            if i[0] <= zuv <= i[1]:
                interval_u = [i]
                break

        Xtilde_u = Omega_u.dot(X)

        while zuv <= interval_u[0][1]:
            Y_zuv = a + b * zuv
            Ytilde_u_zuv = Omega_u.dot(Y_zuv)

            M_v, Mc_v, sM_v = util.solve_Lasso(Xtilde_u, Ytilde_u_zuv, Lambda)

            interval_v = conditioning.conditioning_Zv_Lasso(Xtilde_u, a, b, Lambda, Omega_u, M_v, Mc_v, sM_v)
            
            interval_uv = util.intersect(interval_u, interval_v)

            list_intervals += interval_uv
            list_M += [M_v]
            zuv = interval_uv[0][1] + 1e-4

    return list_intervals, list_M

def SFS_DA_Lasso(Xs, Ys, Xt, Yt, Lambda, Sigma_s, Sigma_t, zmin=-20, zmax=20):
    ns = Ys.shape[0]
    nt = Yt.shape[0]
    
    X = np.vstack((Xs, Xt))
    Y = np.vstack((Ys, Yt))
    Sigma = np.vstack((np.hstack((Sigma_s, np.zeros((ns, nt)))), np.hstack((np.zeros((nt, ns)), Sigma_t))))

    M_obs = util.Lasso_after_DA(X, Y, ns, nt, Lambda)[0]
    k = M_obs.shape[0]

    if k == 0:
        return None

    Xt_M = Xt[:, M_obs]

    list_p_sel = []

    for j in range(k):
        ej = np.zeros((k, 1))
        ej[j][0] = 1

        etaj = np.vstack((np.zeros((ns, 1)), Xt_M.dot(np.linalg.inv(Xt_M.T.dot(Xt_M))).dot(ej)))
        
        b = Sigma.dot(etaj).dot(np.linalg.inv(etaj.T.dot(Sigma).dot(etaj)))
        a = (np.identity(ns + nt) - b.dot(etaj.T)).dot(Y)

        list_intervals, list_M = divide_and_conquer_Lasso(X, a, b, ns, nt, Lambda, zmin, zmax)

        Z = []
        for i in range(len(list_intervals)):
            if np.array_equal(list_M[i], M_obs):
                Z.append(list_intervals[i])

        # Compute p-value
        etajTY = etaj.T.dot(Y)[0][0]
        etajT_Sigma_etaj = etaj.T.dot(Sigma).dot(etaj)[0][0]

        pj_sel = pivot.compute_TN_p_value(Z, etajTY, etajT_Sigma_etaj, 0)

        list_p_sel.append([pj_sel, M_obs[j]])

    return list_p_sel

def SFS_DA_Lasso_rand_feature(Xs, Ys, Xt, Yt, Lambda, Sigma_s, Sigma_t, zmin=-20, zmax=20):
    ns = Ys.shape[0]
    nt = Yt.shape[0]
    
    X = np.vstack((Xs, Xt))
    Y = np.vstack((Ys, Yt))
    Sigma = np.vstack((np.hstack((Sigma_s, np.zeros((ns, nt)))), np.hstack((np.zeros((nt, ns)), Sigma_t))))

    M_obs = util.Lasso_after_DA(X, Y, ns, nt, Lambda)[0]
    k = M_obs.shape[0]

    if k == 0:
        return None

    Xt_M = Xt[:, M_obs]

    j = np.random.randint(k)
    ej = np.zeros((k, 1))
    ej[j][0] = 1

    etaj = np.vstack((np.zeros((ns, 1)), Xt_M.dot(np.linalg.inv(Xt_M.T.dot(Xt_M))).dot(ej)))
    
    b = Sigma.dot(etaj).dot(np.linalg.inv(etaj.T.dot(Sigma).dot(etaj)))
    a = (np.identity(ns + nt) - b.dot(etaj.T)).dot(Y)

    list_intervals, list_M = divide_and_conquer_Lasso(X, a, b, ns, nt, Lambda, zmin, zmax)

    Z = []
    for i in range(len(list_intervals)):
        if np.array_equal(list_M[i], M_obs):
            Z.append(list_intervals[i])

    # Compute p-value
    etajTY = etaj.T.dot(Y)[0][0]
    etajT_Sigma_etaj = etaj.T.dot(Sigma).dot(etaj)[0][0]

    pj_sel = pivot.compute_TN_p_value(Z, etajTY, etajT_Sigma_etaj, 0)

    return pj_sel

def divide_and_conquer_ElasticNet(X, a, b, ns, nt, Lambda, Gamma, zmin, zmax):
    list_intervals = []
    list_M = []

    zuv = zmin

    while zuv <= zmax:
        Y_zuv = a + b * zuv

        # OT from source to target
        H, h, Theta, c_, cost = util.construct(X, Y_zuv, ns, nt)
        Bu, Bcu, Tu = util.LinearProgram(cost, H, h)
        Tu = Tu.reshape(ns, nt)
        Omega_u = np.hstack((np.zeros((ns + nt, ns)), np.vstack((ns * Tu, np.identity(nt)))))

        interval_u = conditioning.conditioning_Zu(a, b, c_, Theta, Bu, Bcu, H)

        # Select the interval containing the data point that we are currently considering.
        for i in interval_u:
            if i[0] <= zuv <= i[1]:
                interval_u = [i]
                break

        Xtilde_u = Omega_u.dot(X)

        while zuv <= interval_u[0][1]:
            Y_zuv = a + b * zuv
            Ytilde_u_zuv = Omega_u.dot(Y_zuv)

            M_v, Mc_v, sM_v = util.solve_ElasticNet(Xtilde_u, Ytilde_u_zuv, Lambda, Gamma)

            interval_v = conditioning.conditioning_Zv_ElasticNet(Xtilde_u, a, b, Lambda, Gamma, Omega_u, M_v, Mc_v, sM_v)
            
            interval_uv = util.intersect(interval_u, interval_v)

            list_intervals += interval_uv
            list_M += [M_v]
            zuv = interval_uv[0][1] + 1e-4

    return list_intervals, list_M

def SFS_DA_ElasticNet(Xs, Ys, Xt, Yt, Lambda, Gamma, Sigma_s, Sigma_t, zmin=-20, zmax=20):
    ns = Ys.shape[0]
    nt = Yt.shape[0]
    
    X = np.vstack((Xs, Xt))
    Y = np.vstack((Ys, Yt))
    Sigma = np.vstack((np.hstack((Sigma_s, np.zeros((ns, nt)))), np.hstack((np.zeros((nt, ns)), Sigma_t))))

    M_obs = util.EN_after_DA(X, Y, ns, nt, Lambda, Gamma)[0]
    k = M_obs.shape[0]

    if k == 0:
        return None

    Xt_M = Xt[:, M_obs]

    list_p_sel = []

    for j in range(k):
        ej = np.zeros((k, 1))
        ej[j][0] = 1

        etaj = np.vstack((np.zeros((ns, 1)), Xt_M.dot(np.linalg.inv(Xt_M.T.dot(Xt_M))).dot(ej)))
        
        b = Sigma.dot(etaj).dot(np.linalg.inv(etaj.T.dot(Sigma).dot(etaj)))
        a = (np.identity(ns + nt) - b.dot(etaj.T)).dot(Y)

        list_intervals, list_M = divide_and_conquer_ElasticNet(X, a, b, ns, nt, Lambda, Gamma, zmin, zmax)

        Z = []
        for i in range(len(list_intervals)):
            if np.array_equal(list_M[i], M_obs):
                Z.append(list_intervals[i])

        # Compute p-value
        etajTY = etaj.T.dot(Y)[0][0]
        etajT_Sigma_etaj = etaj.T.dot(Sigma).dot(etaj)[0][0]

        pj_sel = pivot.compute_TN_p_value(Z, etajTY, etajT_Sigma_etaj, 0)

        list_p_sel.append([pj_sel, M_obs[j]])

    return list_p_sel

def SFS_DA_ElasticNet_rand_feature(Xs, Ys, Xt, Yt, Lambda, Gamma, Sigma_s, Sigma_t, zmin=-20, zmax=20):
    ns = Ys.shape[0]
    nt = Yt.shape[0]
    
    X = np.vstack((Xs, Xt))
    Y = np.vstack((Ys, Yt))
    Sigma = np.vstack((np.hstack((Sigma_s, np.zeros((ns, nt)))), np.hstack((np.zeros((nt, ns)), Sigma_t))))

    M_obs = util.EN_after_DA(X, Y, ns, nt, Lambda, Gamma)[0]
    k = M_obs.shape[0]

    if k == 0:
        return None

    Xt_M = Xt[:, M_obs]

    j = np.random.randint(k)

    ej = np.zeros((k, 1))
    ej[j][0] = 1

    etaj = np.vstack((np.zeros((ns, 1)), Xt_M.dot(np.linalg.inv(Xt_M.T.dot(Xt_M))).dot(ej)))
    
    b = Sigma.dot(etaj).dot(np.linalg.inv(etaj.T.dot(Sigma).dot(etaj)))
    a = (np.identity(ns + nt) - b.dot(etaj.T)).dot(Y)

    list_intervals, list_M = divide_and_conquer_ElasticNet(X, a, b, ns, nt, Lambda, Gamma, zmin, zmax)

    Z = []
    for i in range(len(list_intervals)):
        if np.array_equal(list_M[i], M_obs):
            Z.append(list_intervals[i])

    # Compute p-value
    etajTY = etaj.T.dot(Y)[0][0]
    etajT_Sigma_etaj = etaj.T.dot(Sigma).dot(etaj)[0][0]

    pj_sel = pivot.compute_TN_p_value(Z, etajTY, etajT_Sigma_etaj, 0)

    return pj_sel