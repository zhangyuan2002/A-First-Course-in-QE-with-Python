# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 23:58:48 2024

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt

def solve_λs(coefs):    
    # Calculate the roots using numpy.roots
    λs = np.roots(coefs)
    
    # Sort the roots for consistency
    return sorted(λs, reverse=True)

def solve_η(λ_1, λ_2, y_neg1, y_neg2):
    # Solve the system of linear equation
    A = np.array([
        [1/λ_1, 1/λ_2],
        [1/(λ_1**2), 1/(λ_2**2)]
    ])
    b = np.array((y_neg1, y_neg2))
    ηs = np.linalg.solve(A, b)
    
    return ηs

def solve_sqrt(σ, coefs, y_neg1, y_neg2, t_max=100):
    # Ensure σ is greater than 1
    if σ <= 1:
        raise ValueError("σ must be greater than 1")
        
    # Characteristic roots
    λ_1, λ_2 = solve_λs(coefs)
    
    # Solve for η_1 and η_2
    η_1, η_2 = solve_η(λ_1, λ_2, y_neg1, y_neg2)

    # Compute the sequence up to t_max
    t = np.arange(t_max + 1)
    y = (λ_1 ** t) * η_1 + (λ_2 ** t) * η_2
    
    # Compute the ratio y_{t+1} / y_t for large t
    sqrt_σ_estimate = (y[-1] / y[-2]) - 1
    
    return sqrt_σ_estimate

# Use σ = 2 as an example
σ = 2

# Encode characteristic equation
coefs = (1, -2, (1 - σ))

# Solve for the square root of σ
sqrt_σ = solve_sqrt(σ, coefs, y_neg1=2, y_neg2=1)

# Calculate the deviation
dev = abs(sqrt_σ-np.sqrt(σ))
print(f"sqrt({σ}) is approximately {sqrt_σ:.5f} (error: {dev:.5f})")



# Compute λ_1, λ_2
λ_1, λ_2 = solve_λs(coefs)
print(f'Roots for the characteristic equation are ({λ_1:.5f}, {λ_2:.5f}))')


# Case 1: η_1, η_2 = (0, 1)
ηs = (0, 1)

# Compute y_{t} and y_{t-1} with t >= 0
y = lambda t, ηs: (λ_1 ** t) * ηs[0] + (λ_2 ** t) * ηs[1]
sqrt_σ = 1 - y(1, ηs) / y(0, ηs)

print(f"For η_1, η_2 = (0, 1), sqrt_σ = {sqrt_σ:.5f}")


# Case 2: η_1, η_2 = (1, 0)
ηs = (1, 0)
sqrt_σ = y(1, ηs) / y(0, ηs) - 1

print(f"For η_1, η_2 = (1, 0), sqrt_σ = {sqrt_σ:.5f}")


def iterate_M(x_0, M, num_steps, dtype=np.float64):
    
    # Eigendecomposition of M
    Λ, V = np.linalg.eig(M)
    V_inv = np.linalg.inv(V)
    
    # Initialize the array to store results
    xs = np.zeros((x_0.shape[0], 
                   num_steps + 1))
    
    # Perform the iterations
    xs[:, 0] = x_0
    for t in range(num_steps):
        xs[:, t + 1] = M @ xs[:, t]
    
    return xs, Λ, V, V_inv

# Define the state transition matrix M
M = np.array([
      [2, -(1 - σ)],
      [1, 0]])

# Initial condition vector x_0
x_0 = np.array([2, 2])

# Perform the iteration
xs, Λ, V, V_inv = iterate_M(x_0, M, num_steps=100)

print(f"eigenvalues:\n{Λ}")
print(f"eigenvectors:\n{V}")
print(f"inverse eigenvectors:\n{V_inv}")


roots = solve_λs((1, -2, (1 - σ)))
print(f"roots: {np.round(roots, 8)}")


# Plotting the eigenvectors
plt.figure(figsize=(8, 8))

plt.quiver(0, 0, V[0, 0], V[1, 0], angles='xy', scale_units='xy', 
           scale=1, color='C0', label=fr'$\lambda_1={np.round(Λ[0], 4)}$')
plt.quiver(0, 0, V[0, 1], V[1, 1], angles='xy', scale_units='xy', 
           scale=1, color='C1', label=fr'$\lambda_2={np.round(Λ[1], 4)}$')

# Annotating the slopes
plt.text(V[0, 0]-0.5, V[1, 0]*1.2, 
         r'slope=$\frac{V_{1,1}}{V_{1,2}}=$'+f'{np.round(V[0, 0] / V[1, 0], 4)}', 
         fontsize=12, color='C0')
plt.text(V[0, 1]-0.5, V[1, 1]*1.2, 
         r'slope=$\frac{V_{2,1}}{V_{2,2}}=$'+f'{np.round(V[0, 1] / V[1, 1], 4)}', 
         fontsize=12, color='C1')

# Adding labels
plt.axhline(0, color='grey', linewidth=0.5, alpha=0.4)
plt.axvline(0, color='grey', linewidth=0.5, alpha=0.4)
plt.legend()

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()

xd_1 = np.array((x_0[0], 
                 V[1,1]/V[0,1] * x_0[0]),
                dtype=np.float64)

# Compute x_{1,0}^*
np.round(V_inv @ xd_1, 8)

xd_2 = np.array((x_0[0], 
                 V[1,0]/V[0,0] * x_0[0]), 
                 dtype=np.float64)

# Compute x_{2,0}^*
np.round(V_inv @ xd_2, 8)


# Simulate with muted λ1 λ2.
num_steps = 10
xs_λ1 = iterate_M(xd_1, M, num_steps)[0]
xs_λ2 = iterate_M(xd_2, M, num_steps)[0]

# Compute ratios y_t / y_{t-1}
ratios_λ1 = xs_λ1[1, 1:] / xs_λ1[1, :-1]
ratios_λ2 = xs_λ2[1, 1:] / xs_λ2[1, :-1] 


# Plot the ratios for y_t / y_{t-1}
fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=500)

# First subplot
axs[0].plot(np.round(ratios_λ1, 6), 
            label=r'$\frac{y_t}{y_{t-1}}$', linewidth=3)
axs[0].axhline(y=Λ[1], color='red', linestyle='--', 
               label='$\lambda_2$', alpha=0.5)
axs[0].set_xlabel('t', size=18)
axs[0].set_ylabel(r'$\frac{y_t}{y_{t-1}}$', size=18)
axs[0].set_title(r'$\frac{y_t}{y_{t-1}}$ after Muting $\lambda_1$', 
                 size=13)
axs[0].legend()

# Second subplot
axs[1].plot(ratios_λ2, label=r'$\frac{y_t}{y_{t-1}}$', 
            linewidth=3)
axs[1].axhline(y=Λ[0], color='green', linestyle='--', 
               label='$\lambda_1$', alpha=0.5)
axs[1].set_xlabel('t', size=18)
axs[1].set_ylabel(r'$\frac{y_t}{y_{t-1}}$', size=18)
axs[1].set_title(r'$\frac{y_t}{y_{t-1}}$ after Muting $\lambda_2$', 
                 size=13)
axs[1].legend()

plt.tight_layout()
plt.show()