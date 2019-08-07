import numpy as np
from numba import njit

@njit
def PxHxPH(A, B, C, D, h):
    """ Eq 2.3 & 2.4 page 6 """

    # P = ( A A^2 A^3 A^4 ... A^n ).T
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    Px = np.kron(np.ones((h,1)),A)
    for r in range(n,Px.shape[0],n):
        Px[r:r+n] = A @ Px[(r-n):(r-n)+n]

    IPx = np.concatenate((np.eye(A.shape[0]),Px))

    # H = (    B  0 ... )
    #     (   AB  B ... )
    #     ( A^2B AB ... )
    Hx = np.zeros((n*h,m*h))
    for r in range(0,Hx.shape[0],n):
        for c in range(0,int((r/n+1)*m),m):
            Hx[r:r+n,c:c+m] = IPx[int(r-c/m*n):int(r-c/m*n)+n] @ B

    # P = ( C CA CA^2 ... CA^n-1 ).T
    P = np.zeros((p*h,n))
    for r in range(0,P.shape[0],p):
        P[r:r+p] = C @ IPx[int(r/p*n):int(r/p*n+n)]

    # H =
    H = np.zeros((p*h,m*h))
    for r in range(0,H.shape[0],p):
        if r == 0:
            H[:p,:m] = D
        else:
            H[r:r+p,:m] = \
                C @ IPx[int((r/p-1)*n):int((r/p-1)*n)+n] @ B
    for c in range(m,H.shape[1],m):
        H[int(c/m*p):,c] = H[:-int(c/m*p),0]

    return Px, Hx, P, H

def predict(Px, Hx, P, H, x0, u):
    x = Px @ x0 + Hx @ u
    y = P @ x0 + H @ u
    return x, y
