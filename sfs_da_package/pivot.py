import numpy as np
from mpmath import mp
mp.dps = 500

def compute_TN_p_value(list_intervals, Test_Statistic, Var, mu_0):
    if len(list_intervals) == 0:
        print('Error no interval')
        return None

    tn_sigma = np.sqrt(Var)

    list_tn_cdf = []
    for interval in list_intervals:
        temp = mp.ncdf((interval[1] - mu_0) / tn_sigma) - mp.ncdf((interval[0] - mu_0) / tn_sigma)
        list_tn_cdf.append(temp)

    numerator = 0
    for i in range(len(list_intervals)):
        interval = list_intervals[i]

        if Test_Statistic > interval[1]:
            numerator += list_tn_cdf[i]
        else:
            numerator += mp.ncdf((Test_Statistic - mu_0) / tn_sigma) - mp.ncdf((interval[0] - mu_0) / tn_sigma)
            break

    denominator = sum(list_tn_cdf)

    if denominator == 0.0:
        print('Numerical error')
        return None

    cdf = float(numerator / denominator)

    p_value = 2 * min(cdf, 1 - cdf)
    return p_value