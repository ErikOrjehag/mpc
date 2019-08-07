import numpy as np
from time import time
import linear_models as lm
import linear_mpc as lmpc

m1 = 10.0
m2 = 5.0

dt = 0.01
h = np.round(20.0/dt).astype(np.int_)

x0 = np.array([
    [0],
    [0],
    [0],
    [0],
])

u = np.zeros((1*h, 1)).astype(np.float_)
u[:5,0] = 0.1

#A, B, C, D = lm.spring_damper(m2, 0.1, 0.01, dt=dt)
A, B, C, D = lm.overhead_crane(m1, m2, dt=dt)

start = time()
Px, Hx, P, H = lmpc.PxHxPH(A, B, C, D, h)
#H = np.nan_to_num(H)
print("PxHxPH took %.5fs" % (time() - start))

start = time()
x, y = lmpc.predict(Px, Hx, P, H, x0, u)
print("predict took %.5fs" % (time() - start))

"""
print(Px)
print(Hx)
print(P)
"""
"""
print(D)
print(C@B)
print(C@A@B)
print(C@A@A@B)
print(C@A@A@A@B)
print(H)
"""
"""
print(x0)
print(x)
print(y)
"""

import matplotlib.pyplot as plt
#plt.plot(x[::2])
plt.plot(y)
plt.show()
