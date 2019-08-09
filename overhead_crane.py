import numpy as np
from time import time
import linear_models as lm
import linear_mpc as lmpc

m1 = 10.0
m2 = 5.0

dt = 0.01
horizon = 1.0 # seconds
h = np.round(horizon/dt).astype(np.int_)

x0 = np.array([
    [0],
    [0],
    [0],
    [0],
])

usize = 1
u = np.zeros((usize*h, 1)).astype(np.float_)
#u[:5,0] = 0.1

Q = lmpc.repeat_diag(h, np.eye(x0.shape[0])*0.5)
R = lmpc.repeat_diag(h, np.eye(usize)*0.5)

#A, B, C, D = lm.spring_damper(m2, 0.1, 0.01, dt=dt)
A, B, C, D = lm.overhead_crane(m1, m2, dt=dt)

start = time()
Px, Hx, P, H = lmpc.PxHxPH(A, B, C, D, h)
#H = np.nan_to_num(H)
print("PxHxPH took: %.5fs" % (time() - start))

start = time()
x, y = lmpc.predict(Px, Hx, P, H, x0, u)
print("predict took: %.5fs" % (time() - start))

cost = lmpc.J(x, u, Q, R)
print("cost is: ", cost)

G, fT, c = lmpc.GfTc(x0, Px, Hx, Q, R)
print(G)
print(fT)
print(c)

uo = lmpc.unconstrained_u(G, fT)

print(uo)

import matplotlib.pyplot as plt
#plt.plot(x[::2])
plt.plot(y)
plt.show()
