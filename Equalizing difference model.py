# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 09:37:40 2024

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sympy import Symbol, Lambda, symbols

# Define the namedtuple for the equalizing difference model
EqDiffModel = namedtuple('EqDiffModel', 'R T γ_h γ_c w_h0 D')

def create_edm(R=1.05,   # gross rate of return
               T=40,     # time horizon
               γ_h=1.01, # high-school wage growth
               γ_c=1.01, # college wage growth
               w_h0=1,   # initial wage (high school)
               D=10,     # cost for college
              ):
    
    return EqDiffModel(R, T, γ_h, γ_c, w_h0, D)

def compute_gap(model):
    R, T, γ_h, γ_c, w_h0, D = model
    
    A_h = (1 - (γ_h/R)**(T+1)) / (1 - γ_h/R)
    A_c = (1 - (γ_c/R)**(T-3)) / (1 - γ_c/R) * (γ_c/R)**4
    ϕ = A_h / A_c + D / (w_h0 * A_c)
    
    return ϕ


ex1 = create_edm()
gap1 = compute_gap(ex1)

gap1

# free college
ex2 = create_edm(D=0)
gap2 = compute_gap(ex2)
gap2

R_arr = np.linspace(1, 1.2, 50)
models = [create_edm(R=r) for r in R_arr]
gaps = [compute_gap(model) for model in models]

plt.plot(R_arr, gaps)
plt.xlabel(r'$R$')
plt.ylabel(r'wage gap')
plt.show()


γc_arr = np.linspace(1, 1.2, 50)
models = [create_edm(γ_c=γ_c) for γ_c in γc_arr]
gaps = [compute_gap(model) for model in models]

plt.plot(γc_arr, gaps)
plt.xlabel(r'$\gamma_c$')
plt.ylabel(r'wage gap')
plt.show()


γh_arr = np.linspace(1, 1.1, 50)
models = [create_edm(γ_h=γ_h) for γ_h in γh_arr]
gaps = [compute_gap(model) for model in models]

plt.plot(γh_arr, gaps)
plt.xlabel(r'$\gamma_h$')
plt.ylabel(r'wage gap')
plt.show()


# Define a model of entrepreneur-worker interpretation
EqDiffModel = namedtuple('EqDiffModel', 'R T γ_h γ_c w_h0 D π')

def create_edm_π(R=1.05,   # gross rate of return
                 T=40,     # time horizon
                 γ_h=1.01, # high-school wage growth
                 γ_c=1.01, # college wage growth
                 w_h0=1,   # initial wage (high school)
                 D=10,     # cost for college
                 π=0       # chance of business success
              ):
    
    return EqDiffModel(R, T, γ_h, γ_c, w_h0, D, π)


def compute_gap(model):
    R, T, γ_h, γ_c, w_h0, D, π = model
    
    A_h = (1 - (γ_h/R)**(T+1)) / (1 - γ_h/R)
    A_c = (1 - (γ_c/R)**(T-3)) / (1 - γ_c/R) * (γ_c/R)**4
    
    # Incorprate chance of success
    A_c = π * A_c
    
    ϕ = A_h / A_c + D / (w_h0 * A_c)
    return ϕ


ex3 = create_edm_π(π=0.2)
gap3 = compute_gap(ex3)

gap3


π_arr = np.linspace(0.2, 1, 50)
models = [create_edm_π(π=π) for π in π_arr]
gaps = [compute_gap(model) for model in models]

plt.plot(π_arr, gaps)
plt.ylabel(r'wage gap')
plt.xlabel(r'$\pi$')
plt.show()


γ_h, γ_c, w_h0, D = symbols('\gamma_h, \gamma_c, w_0^h, D', real=True)
R, T = Symbol('R', real=True), Symbol('T', integer=True)

A_h = Lambda((γ_h, R, T), (1 - (γ_h/R)**(T+1)) / (1 - γ_h/R))
A_h

A_c = Lambda((γ_c, R, T), (1 - (γ_c/R)**(T-3)) / (1 - γ_c/R) * (γ_c/R)**4)
A_c

ϕ = Lambda((D, γ_h, γ_c, R, T, w_h0), A_h(γ_h, R, T)/A_c(γ_c, R, T) + D/(w_h0*A_c(γ_c, R, T)))
ϕ


R_value = 1.05
T_value = 40
γ_h_value, γ_c_value = 1.01, 1.01
w_h0_value = 1
D_value = 10

ϕ_D = ϕ(D, γ_h, γ_c, R, T, w_h0).diff(D)
ϕ_D

# Numerical value at default parameters
ϕ_D_func = Lambda((D, γ_h, γ_c, R, T, w_h0), ϕ_D)
ϕ_D_func(D_value, γ_h_value, γ_c_value, R_value, T_value, w_h0_value)

ϕ_T = ϕ(D, γ_h, γ_c, R, T, w_h0).diff(T)
ϕ_T

# Numerical value at default parameters
ϕ_T_func = Lambda((D, γ_h, γ_c, R, T, w_h0), ϕ_T)
ϕ_T_func(D_value, γ_h_value, γ_c_value, R_value, T_value, w_h0_value)


ϕ_γ_h = ϕ(D, γ_h, γ_c, R, T, w_h0).diff(γ_h)
ϕ_γ_h


# Numerical value at default parameters
ϕ_γ_h_func = Lambda((D, γ_h, γ_c, R, T, w_h0), ϕ_γ_h)
ϕ_γ_h_func(D_value, γ_h_value, γ_c_value, R_value, T_value, w_h0_value)


ϕ_γ_c = ϕ(D, γ_h, γ_c, R, T, w_h0).diff(γ_c)
ϕ_γ_c

# Numerical value at default parameters
ϕ_γ_c_func = Lambda((D, γ_h, γ_c, R, T, w_h0), ϕ_γ_c)
ϕ_γ_c_func(D_value, γ_h_value, γ_c_value, R_value, T_value, w_h0_value)


ϕ_R = ϕ(D, γ_h, γ_c, R, T, w_h0).diff(R)
ϕ_R
# Numerical value at default parameters
ϕ_R_func = Lambda((D, γ_h, γ_c, R, T, w_h0), ϕ_R)
ϕ_R_func(D_value, γ_h_value, γ_c_value, R_value, T_value, w_h0_value)
