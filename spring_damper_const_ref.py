import numpy as np
from time import time
import linear_models as lm
import linear_mpc as lmpc

m = 1.0 # mass
k = 2.0 # spring
c = 0.005 # damper

dt = 0.01
horizon = 12.0 # seconds
h = np.round(horizon/dt).astype(np.int_)
t = np.arange(0, horizon, dt)

x0 = np.array([
    [1.0], # position
    [0], # velocity
])
yr = np.array([[0.75]])

A, B, C, D = lm.spring_damper(m, k, c, dt)
Ar, Br, Cr, Ce, x0 = lmpc.constant_reference(A, B, C, x0, yr)

usize = 1
u = np.zeros((usize*h, 1)).astype(np.float_)

# # # #

Pxr, Hxr, Pr, Hr = lmpc.PxHxPH(Ar, Br, Cr, D, h)
x, y = lmpc.predict(Pxr, Hxr, Pr, Hr, x0, u)

# # # #

start = time()
Pxe, Hxe, Pe, He = lmpc.PxHxPH(Ar, Br, Ce, D, h)
print("PxHxPH took: %.5fs" % (time() - start))

Q = lmpc.repeat_diag(h, np.eye(x0.shape[0])*1.0)
R = lmpc.repeat_diag(h, np.eye(usize)*0.1)

start = time()
xe, ye = lmpc.predict(Pxe, Hxe, Pe, He, x0, u)
print("predict took: %.5fs" % (time() - start))

j = lmpc.J(xe, u, Q, R)
print("original cost: ", j)

G, fT, c = lmpc.GfTc(x0, Pxe, Hxe, Q, R)

uo = lmpc.unconstrained_u(G, fT)

xo, yo = lmpc.predict(Pxr, Hxr, Pr, Hr, x0, uo)

jo = lmpc.J(xo, uo, Q, R)
print("optimized cost is: ", jo)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t, y)
ax[0].plot(t, yo)
ax[0].plot(t, ye)
ax[0].set_ylabel("position")
ax[1].plot(t[:uo.shape[0]], uo, 'r')
ax[1].set_ylabel("control force")
ax[1].set_xlabel("seconds")
plt.show()
