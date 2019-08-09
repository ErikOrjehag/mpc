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

A, B, C, D = lm.spring_damper(m, k, c, dt)

start = time()
Px, Hx, P, H = lmpc.PxHxPH(A, B, C, D, h)
print("PxHxPH took: %.5fs" % (time() - start))
# Move blocking
usize = 1
ctrl = h//3
Hx, H = lmpc.move_block(h, ctrl, usize, Hx, H)

u = np.zeros((usize*ctrl, 1)).astype(np.float_)

Q = lmpc.repeat_diag(h, np.eye(x0.shape[0])*1.0)
R = lmpc.repeat_diag(ctrl, np.eye(usize)*0.5)

start = time()
x, y = lmpc.predict(Px, Hx, P, H, x0, u)
print("predict took: %.5fs" % (time() - start))

j = lmpc.J(x, u, Q, R)
print("cost is: ", j)

G, fT, c = lmpc.GfTc(x0, Px, Hx, Q, R)
#print(G)
#print(fT)
#print(c)

uu = lmpc.unconstrained_u(G, fT)
xx, yy = lmpc.predict(Px, Hx, P, H, x0, uu)
jj = lmpc.J(xx, uu, Q, R)
print("cost is: ", jj)

#print(uo)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t, y)
ax[0].plot(t, yy)
ax[0].set_ylabel("position")
uuu = np.zeros((h,1))
uuu[:uu.shape[0]] = uu
ax[1].plot(t, uuu, 'r')
ax[1].set_ylabel("control force")
ax[1].set_xlabel("seconds")
plt.show()
