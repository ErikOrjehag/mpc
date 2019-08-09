import numpy as np
from time import time
import linear_models as lm
import linear_mpc as lmpc

m = 1.0 # mass
k = 2.0 # spring
c = 0.005 # damper

dt = 0.01
horizon = 10.0 # seconds
h = np.round(horizon/dt).astype(np.int_)
t = np.arange(0, horizon, dt)

x0 = np.array([
    [1.0], # position
    [0], # velocity
])

usize = 1
u = np.zeros((usize*h, 1)).astype(np.float_)

Q = lmpc.repeat_diag(h, np.eye(x0.shape[0])*1.0)
R = lmpc.repeat_diag(h, np.eye(usize)*0.5)

A, B, C, D = lm.spring_damper(m, k, c, dt)

start = time()
Px, Hx, P, H = lmpc.PxHxPH(A, B, C, D, h)
print("PxHxPH took: %.5fs" % (time() - start))

start = time()
x, y = lmpc.predict(Px, Hx, P, H, x0, u)
print("predict took: %.5fs" % (time() - start))

cost = lmpc.J(x, u, Q, R)
print("cost is: ", cost)

G, fT, c = lmpc.GfTc(x0, Px, Hx, Q, R)
#print(G)
#print(fT)
#print(c)

uu = lmpc.unconstrained_u(G, fT)
_, yy = lmpc.predict(Px, Hx, P, H, x0, uu)

#print(uo)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t, y)
ax[0].plot(t, yy)
ax[0].set_ylabel("position")
ax[1].plot(t, uu, 'r')
ax[1].set_ylabel("control force")
ax[1].set_xlabel("seconds")
plt.show()
