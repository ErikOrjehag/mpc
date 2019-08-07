import numpy as np

def as_float(*args):
    return tuple([arg.astype(np.float_) for arg in args])

def spring_damper(m, k, c, dt=0.01):
    A = np.array([
        [1      , dt     ],
        [-k/m*dt, 1 -c/m ],
    ])
    B = np.array([
        [0     ],
        [1/m*dt],
    ])
    C = np.array([
        [1, 0],
    ])
    D = np.array([
        [0],
    ])
    return as_float(A, B, C, D)

def overhead_crane(m1, m2, dt=0.01):
    A = np.array([
        [1, dt, 0            , 0 ],
        [0,  1, m2/(m1+m2)*dt, 0 ],
        [0,  0, 1            , dt],
        [0,  0, -dt          , 1 ],
    ])
    B = np.array([
        [ 0 ],
        [ dt],
        [ 0 ],
        [-dt],
    ])
    C = np.array([
        [1,0,0,0],
    ])
    D = np.array([
        [0],
    ])
    return as_float(A, B, C, D)
