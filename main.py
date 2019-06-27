import numpy as np

def PxHxPH(A, B, C, D, n):
    """ Eq 2.3 & 2.4 page 6 """
    # P = ( A A^2 A^3 A^4 ... A^n ).T
    Px = np.full((n,*A.shape), A)
    Px = np.array([ Px[0] ] + [ Px[i]@Px[i+1] for i in range(0,n-1) ])

    # H = (    B  0 ... )
    #     (   AB  B ... )
    #     ( A^2B AB ... )
    IPx = np.concatenate((np.eye(A.shape[0])[np.newaxis],Px))
    Hx = np.empty((n,n,*B.shape))
    for r in range(Hx.shape[0]):
        for c in range(0,min(r+1,Hx.shape[1])):
            Hx[r,c] = IPx[r-c] @ B

    # P = ( C CA CA^2 ... CA^n-1 ).T
    P = np.empty((n,*C.shape))
    for r in range(P.shape[0]):
        P[r] = C @ IPx[r]

    # H =
    H = np.empty((n,n,*D.shape))
    for r in range(Hx.shape[0]):
        for c in range(0,min(r+1,Hx.shape[1])):
            if r == c:
                H[r,c] = D
            else:
                H[r,c] = C @ IPx[r-1-c] @ B

    return Px, Hx, P, H

def predict(Px, Hx, P, H, x0, u):
    print(Hx, u)
    print(Hx.shape, u.shape)
    print(Hx @ u)
    x = Px @ x0 + Hx @ u
    y = P @ x0 + H @ u
    return x, y

x0 = np.zeros((3,1))
A = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
B = np.array([
    [3, 6],
    [2, 5],
    [1, 4]
])
C = np.array([
    [2, 3, 4],
    [5, 6, 7]
])
D = np.array([
    [6, 7],
    [8, 9]
])

n = 3

Px, Hx, P, H = PxHxPH(A,B, C, D, n)

x0 = np.array([[1], [2], [3]])
u = np.repeat(np.zeros((2, 1))[np.newaxis], n, axis=0)

x, y = predict(Px, Hx, P, H, x0, u)

print(x)
print(y)
