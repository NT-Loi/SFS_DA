import numpy as np
import util

def solve_linear_unequalities(left, right):
    # Solve unequality has the form of ax >= b
    
    Z_minus = np.NINF
    Z_plus = np.inf

    for i in range(left.shape[0]):
        l = left[i][0]
        r = right[i][0]

        if - 1e-10 <= l <= 1e-10:
            l = 0.0

        if l == 0.0:
            continue

        else:
            temp = r / l

            if l > 0.0:
                Z_minus = max(temp, Z_minus)
            else:
                Z_plus = min(temp, Z_plus)

    return [[Z_minus, Z_plus]]

def solve_quadratic_inequality(a, b, c):
    # Solve quadratic inequality has the form of ax^2 + bx + c <= 0
    if -1e-7 <= a <= 1e-7:
        a = 0.0

    if -1e-7 <= b <= 1e-7:
        b = 0.0

    if -1e-7 <= c <= 1e-7:
        c = 0.0

    if a == 0.0:
        if b != 0.0:
            root = np.round(-c/b, 12)

            if b > 0.0:
                return [[np.NINF, root]]
            else:
                return [[root, np.inf]]
        
        else:
            if c > 0.0:
                # print("Error no roots: c > 0")
                return None
            else:
                return [[np.NINF, np.inf]]
            
    delta = b**2 - 4 * a * c

    if delta < 0.0:
        if a > 0.0:
            # print("Error no roots: a > 0")
            return None
        else: 
            return [[np.NINF, np.inf]]
    else:
        sqrt_delta = np.sqrt(delta)

        if b > 0:
            root1 = np.round((-b-sqrt_delta)/(2*a), 12)
        else:
            root1 = np.round((-b+sqrt_delta)/(2*a), 12)

        root2 = np.round(c / (a * root1), 12)

        roots = np.sort([root1, root2])
        
        if a > 0:
            return [roots]
        else:
            return [[np.NINF, roots[0]], [roots[1], np.inf]]
            
def conditioning_Zu(a, b, c_, Theta, B, Bc, H):
    Theta_a = Theta.dot(a)
    Theta_b = Theta.dot(b)

    p_tilde = c_ + Theta_a * Theta_a
    q_tilde = Theta_a * Theta_b + Theta_b * Theta_a
    r_tilde = Theta_b * Theta_b

    H_B_invH_Bc = np.linalg.inv(H[:, B]).dot(H[:, Bc])

    p = (p_tilde[Bc, :].T - p_tilde[B, :].T.dot(H_B_invH_Bc)).T
    q = (q_tilde[Bc, :].T - q_tilde[B, :].T.dot(H_B_invH_Bc)).T
    r = (r_tilde[Bc, :].T - r_tilde[B, :].T.dot(H_B_invH_Bc)).T

    flag = False
    list_intervals = []

    for i in range(p.shape[0]):
        fa = - r[i][0]
        sa = - q[i][0]
        ta = - p[i][0]

        temp = solve_quadratic_inequality(fa, sa, ta)
        
        if flag == False:
            flag = True
            list_intervals = temp
        else:
            list_intervals = util.intersect(list_intervals, temp)

    return list_intervals

def conditioning_Zv_Lasso(Xtilde, a, b, Lambda, Omega, M, Mc, sM):
    XtildeM = Xtilde[:, M]
    XtildeMc = Xtilde[:, Mc]

    XtildeMT_XtildeM_inv = np.linalg.inv(XtildeM.T.dot(XtildeM))
    XtildeM_plus = XtildeMT_XtildeM_inv.dot(XtildeM.T)
    XtildeMT_plus = XtildeM.dot(XtildeMT_XtildeM_inv)

    # Construct Atilde0 and Btilde0
    Atilde0 = - sM * (XtildeM_plus).dot(Omega) 
    Btilde0 = - Lambda * sM * (XtildeMT_XtildeM_inv.dot(sM))

    # Construct Atilde1 and Btilde1
    Atilde10 = XtildeMc.T.dot(np.identity(Xtilde.shape[0]) - XtildeM.dot(XtildeM_plus)).dot(Omega)
    Atilde11 = - Atilde10

    tmp = XtildeMc.T.dot(XtildeMT_plus).dot(sM)
    Btilde10 = np.ones((tmp.shape[0], 1)) - tmp
    Btilde11 = np.ones((tmp.shape[0], 1)) + tmp

    Atilde1 = np.vstack((Atilde10, Atilde11)) / Lambda
    Btilde1 = np.vstack((Btilde10, Btilde11))

    Atilde = np.vstack((Atilde0, Atilde1))
    Btilde = np.vstack((Btilde0, Btilde1))

    A = Atilde.dot(b)
    B = Btilde - Atilde.dot(a)
    return solve_linear_unequalities(-A, -B)

def conditioning_Zv_ElasticNet(Xtilde, a, b, Lambda, Gamma, Omega, M, Mc, sM):
    XtildeM = Xtilde[:, M]
    XtildeMc = Xtilde[:, Mc]

    XtildeMT_XtildeM_inv = np.linalg.inv(XtildeM.T.dot(XtildeM) + Gamma * np.identity(XtildeM.shape[1]))
    XtildeM_star = XtildeMT_XtildeM_inv.dot(XtildeM.T)
    XtildeMT_star = XtildeM.dot(XtildeMT_XtildeM_inv)

    # Construct Atilde0 and Btilde0
    Atilde0 = - sM * (XtildeM_star).dot(Omega) 
    Btilde0 = - Lambda * sM * (XtildeMT_XtildeM_inv.dot(sM))

    # Construct Atilde1 and Btilde1
    Atilde10 = XtildeMc.T.dot(np.identity(Xtilde.shape[0]) - XtildeM.dot(XtildeM_star)).dot(Omega)
    Atilde11 = - Atilde10

    tmp = XtildeMc.T.dot(XtildeMT_star).dot(sM)
    Btilde10 = np.ones((tmp.shape[0], 1)) - tmp
    Btilde11 = np.ones((tmp.shape[0], 1)) + tmp

    Atilde1 = np.vstack((Atilde10, Atilde11)) / Lambda
    Btilde1 = np.vstack((Btilde10, Btilde11))

    Atilde = np.vstack((Atilde0, Atilde1))
    Btilde = np.vstack((Btilde0, Btilde1))

    A = Atilde.dot(b)
    B = Btilde - Atilde.dot(a)
    return solve_linear_unequalities(-A, -B)