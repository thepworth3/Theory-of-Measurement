# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:42:27 2024

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from scipy.stats import binom

# Define constants for the problem
n_bacteria = 10  # total bacteria in the first sample
n_type_a = 3     # type A bacteria observed
n_trials = 12    # total bacteria in the second sample


# Calculate the binomial coefficient for the likelihood
binomial_coeff = np.math.comb(n_bacteria, n_type_a)

# Define the unnormalized posterior function Q^3 * (1-Q)^7
def unnormalized_posterior(Q):
    return binomial_coeff*(Q**3 * (1-Q)**7)

# Calculate the normalization factor p(D | M, I)
normalization_factor, _ = quad(unnormalized_posterior, 0, 1) #quad integrates function from 0 to 1 wrt W
normalization_factor *= binomial_coeff

# Define the normalized posterior distribution
def normalized_posterior(Q):
    return binomial_coeff * unnormalized_posterior(Q) / normalization_factor

# Q values for plotting
Q = np.linspace(0, 1, 1000)

# Calculate the posterior distribution values
posterior_vals = normalized_posterior(Q)

# Plotting the normalized posterior distribution for part (a)
plt.figure(figsize=(10, 6))
plt.plot(Q, posterior_vals, label='Normalized Posterior Distribution', linewidth=2)
plt.title('Normalized Posterior Distribution $p(Q|D, M, I)$')
plt.xlabel('Probability Q of bacteria being type A')
plt.ylabel('Density')
plt.grid(True)
plt.legend()
plt.show()

norm_test = quad(normalized_posterior, 0, 1) #check the PDF is normalized
print(norm_test)


# Part (b): Probability of obtaining exactly 6 type A bacteria in 12 trials
n_success = 6  # 6 type A bacteria in the second sample

# Define the integrand for part (b)
def integrand_6_in_12(Q):
    return binom.pmf(n_success, n_trials, Q) * normalized_posterior(Q)

# Calculate the probability of exactly 6 type A bacteria in 12 trials
prob_6_in_12, _ = quad(integrand_6_in_12, 0, 1)

# Part (c): Probability of obtaining at least 3 type A bacteria in 12 trials
def integrand_at_least_3_in_12(Q):
    prob_at_least_3 = sum(binom.pmf(k, n_trials, Q) for k in range(3, n_trials ))
    return prob_at_least_3 * normalized_posterior(Q)

# Calculate the probability of at least 3 type A bacteria in 12 trials
prob_at_least_3_in_12, _ = quad(integrand_at_least_3_in_12, 0, 1)

# Display results for parts (b) and (c)
print("Probability of exactly 6 type A bacteria in 12 trials:", prob_6_in_12)
print("Probability of at least 3 type A bacteria in 12 trials:", prob_at_least_3_in_12)