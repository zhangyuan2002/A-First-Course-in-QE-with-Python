# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:38:38 2024

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
# Set the axes through the origin
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position('zero')
for spine in ['right', 'top']:
    ax.spines[spine].set_color('none')

ax.set(xlim=(-5, 5), ylim=(-5, 5))

vecs = ((2, 4), (-3, 3), (-4, -3.5))
for v in vecs:
    ax.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(facecolor='blue',
                shrink=0,
                alpha=0.7,
                width=0.5))
    ax.text(1.1 * v[0], 1.1 * v[1], str(v))
plt.show()

fig, ax = plt.subplots()
# Set the axes through the origin
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position('zero')
for spine in ['right', 'top']:
    ax.spines[spine].set_color('none')

ax.set(xlim=(-2, 10), ylim=(-4, 4))
# ax.grid()
vecs = ((4, -2), (3, 3), (7, 1))
tags = ('(x1, x2)', '(y1, y2)', '(x1+x2, y1+y2)')
colors = ('blue', 'green', 'red')
for i, v in enumerate(vecs):
    ax.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(color=colors[i],
                shrink=0,
                alpha=0.7,
                width=0.5,
                headwidth=8,
                headlength=15))
    ax.text(v[0] + 0.2, v[1] + 0.1, tags[i])

for i, v in enumerate(vecs):
    ax.annotate('', xy=(7, 1), xytext=v,
                arrowprops=dict(color='gray',
                shrink=0,
                alpha=0.3,
                width=0.5,
                headwidth=5,
                headlength=20))
plt.show()

fig, ax = plt.subplots()
# Set the axes through the origin
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position('zero')
for spine in ['right', 'top']:
    ax.spines[spine].set_color('none')

ax.set(xlim=(-5, 5), ylim=(-5, 5))
x = (2, 2)
ax.annotate('', xy=x, xytext=(0, 0),
            arrowprops=dict(facecolor='blue',
            shrink=0,
            alpha=1,
            width=0.5))
ax.text(x[0] + 0.4, x[1] - 0.2, '$x$', fontsize='16')

scalars = (-2, 2)
x = np.array(x)

for s in scalars:
    v = s * x
    ax.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(facecolor='red',
                shrink=0,
                alpha=0.5,
                width=0.5))
    ax.text(v[0] + 0.4, v[1] - 0.2, f'${s} x$', fontsize='16')
plt.show()



x = np.ones(3)            # Vector of three ones
y = np.array((2, 4, 6))   # Converts tuple (2, 4, 6) into a NumPy array
x + y                     # Add (element-by-element)

4 * x                     # Scalar multiply


np.sum(x*y)      # Inner product of x and y

x @ y            # Another way to compute the inner product 

np.sqrt(np.sum(x**2))  # Norm of x, method one

np.linalg.norm(x)      # Norm of x, method two

A = ((1, 2),
     (3, 4))

type(A)

A = np.array(A)

type(A)

A.shape

A = np.identity(3)    # 3 x 3 identity matrix
B = np.ones((3, 3))   # 3 x 3 matrix of ones
2 * A

A + B

fig, ax = plt.subplots()
x = np.linspace(-10, 10)
plt.plot(x, (3-x)/3, label=f'$x + 3y = 3$')
plt.plot(x, (-8-2*x)/6, label=f'$2x + 6y = -8$')
plt.legend()
plt.show()


C = ((10, 5),      # Matrix C
     (5, 10))

C = np.array(C)
D = ((-10, -5),     # Matrix D
     (-1, -10))
D = np.array(D)
h = np.array((100, 50))   # Vector h
h.shape = 2,1             # Transforming h to a column vector
from numpy.linalg import det, inv
A = C - D
# Check that A is nonsingular (non-zero determinant), and hence invertible
det(A)

A_inv = inv(A)  # compute the inverse
A_inv

p = A_inv @ h  # equilibrium prices
p

q = C @ p  # equilibrium quantities
q

from numpy.linalg import solve
p = solve(A, h)  # equilibrium prices
p

q = C @ p  # equilibrium quantities
q