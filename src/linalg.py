import numpy as np

def solve_tridiagonal(a, b, c, d):
    """
    Solve tridiagonal system:
    a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    where a[0]=0 and c[-1]=0.
    """
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x