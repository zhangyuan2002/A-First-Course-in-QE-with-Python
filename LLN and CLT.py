# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:53:18 2024

@author: Admin
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

p = 0.8
X = st.bernoulli.rvs(p)
print(X)

n = 1_000_000
X_draws = st.bernoulli.rvs(p, size=n)
print(X_draws.mean()) # count the number of 1's and divide by n

p = 0.3
X_draws = st.bernoulli.rvs(p, size=n)
print(X_draws.mean())


def draw_means(X_distribution,  # The distribution of each X_i
               n):              # The size of the sample mean

    # Generate n draws: X_1, ..., X_n
    X_samples = X_distribution.rvs(size=n)

    # Return the sample mean
    return np.mean(X_samples)

def generate_histogram(X_distribution, n, m): 

    # Compute m sample means

    sample_means = np.empty(m)
    for j in range(m):
      sample_means[j] = draw_means(X_distribution, n) 

    # Generate a histogram

    fig, ax = plt.subplots()
    ax.hist(sample_means, bins=30, alpha=0.5, density=True)
    μ = X_distribution.mean()  # Get the population mean
    σ = X_distribution.std()    # and the standard deviation
    ax.axvline(x=μ, ls="--", c="k", label=fr"$\mu = {μ}$")
     
    ax.set_xlim(μ - σ, μ + σ)
    ax.set_xlabel(r'$\bar X_n$', size=12)
    ax.set_ylabel('density', size=12)
    ax.legend()
    plt.show()
    
# pick a distribution to draw each $X_i$ from
X_distribution = st.norm(loc=5, scale=2) 
# Call the function
generate_histogram(X_distribution, n=1_000, m=1000)


def means_violin_plot(distribution,  
                      ns = [1_000, 10_000, 100_000],
                      m = 10_000):

    data = []
    for n in ns:
        sample_means = [draw_means(distribution, n) for i in range(m)]
        data.append(sample_means)

    fig, ax = plt.subplots()

    ax.violinplot(data)
    μ = distribution.mean()
    ax.axhline(y=μ, ls="--", c="k", label=fr"$\mu = {μ}$")

    labels=[fr'$n = {n}$' for n in ns]

    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)


    plt.subplots_adjust(bottom=0.15, wspace=0.05)

    ax.set_ylabel('density', size=12)
    ax.legend()
    plt.show()
    
means_violin_plot(st.norm(loc=5, scale=2))

means_violin_plot(st.beta(6, 6))

# Set parameters
n = 250         # Choice of n
k = 1_000_000        # Number of draws of Y_n
distribution = st.expon(2) # Exponential distribution, λ = 1/2
μ, σ = distribution.mean(), distribution.std()

# Draw underlying RVs. Each row contains a draw of X_1,..,X_n
data = distribution.rvs((k, n))
# Compute mean of each row, producing k draws of \bar X_n
sample_means = data.mean(axis=1)
# Generate observations of Y_n
Y = np.sqrt(n) * (sample_means - μ)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
xmin, xmax = -3 * σ, 3 * σ
ax.set_xlim(xmin, xmax)
ax.hist(Y, bins=60, alpha=0.4, density=True)
xgrid = np.linspace(xmin, xmax, 200)
ax.plot(xgrid, st.norm.pdf(xgrid, scale=σ), 
        'k-', lw=2, label='$N(0, \sigma^2)$')
ax.set_xlabel(r"$Y_n$", size=12)
ax.set_ylabel(r"$density$", size=12)

ax.legend()

plt.show()