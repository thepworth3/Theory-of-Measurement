# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:53:40 2024

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Raw data (from the given problem)
data = np.array([
    0.25, -0.19, 0.25, -0.56, -0.41, -0.94, 0.84, -0.30, -2.06, -1.39, 
    0.07, 1.80, -1.02, -0.46, 0.29, -0.36, -0.42, 1.43, -1.33, 0.06, 
    0.82, 0.42, 3.76, 1.10, 1.31, 1.86, 0.32, -1.14, 1.24, -0.29, 
    0.02, -1.52, 0.44, 0.05, 0.59, 0.94, -0.10, 0.57, 0.40, -0.97, 
    2.20, 0.15, -0.37, -0.67, -0.05, -0.20, 0.65, -1.24, -1.56, -0.64, 
    0.48, 1.79, 0.07, 1.30, 0.29, -0.23, -0.50, 0.93, -1.28, -1.98, 
    1.85, 0.89, 0.65, 0.28
])

# Step 2: Constants for the problem
sigma_noise = 1.0  # Noise level (mK)
sigma_L = 2  # Line width in channels
predicted_channel = 24  # Center channel of the spectral line
S_values = np.linspace(0.1, 100, 500)  # Range of signal strengths (mK)

# Step 3: Defining functions for likelihood and priors

# Gaussian model for the spectral line
def gaussian_spectral_line(S, channel, predicted_channel, sigma_L):
    return S * np.exp(-((channel - predicted_channel) ** 2) / (2 * sigma_L ** 2))

# Likelihood function: Gaussian likelihood based on the given data
def likelihood(S, data, predicted_channel, sigma_L, sigma_noise):
    model = gaussian_spectral_line(S, np.arange(1, 65), predicted_channel, sigma_L)
    return np.exp(-0.5 * np.sum(((data - model) / sigma_noise) ** 2))

# Jeffreys prior: proportional to 1/S
def jeffreys_prior(S):
    return 1 / S if S > 0 else 0

# Uniform prior: constant between 0.1 and 100 mK
def uniform_prior(S, S_min=0.1, S_max=100):
    return 1 / (S_max - S_min) if S_min <= S <= S_max else 0

# Step 4: Compute the posterior probability for both priors

# Compute posterior probabilities for Jeffreys and Uniform priors
jeffreys_posterior = [likelihood(S, data, predicted_channel, sigma_L, sigma_noise) * jeffreys_prior(S) for S in S_values]
uniform_posterior = [likelihood(S, data, predicted_channel, sigma_L, sigma_noise) * uniform_prior(S) for S in S_values]

# Normalize the posteriors
jeffreys_posterior /= np.sum(jeffreys_posterior)
uniform_posterior /= np.sum(uniform_posterior)

# Step 5: Plot the posterior probabilities (PDF) for both priors
plt.figure(figsize=(10, 6))
plt.plot(S_values, jeffreys_posterior, label="Jeffreys Prior (1/S)", color='b')
plt.plot(S_values, uniform_posterior, label="Uniform Prior", color='r')
plt.title('Posterior Probability Density Function (PDF) for Signal Strength (Theory 1)')
plt.xlabel('Signal Strength (mK)')
plt.ylabel('Posterior Probability')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Compute the most probable value and the 95% credible region for both priors

# Sort posterior values in descending order
sorted_indices_jeffreys = np.argsort(jeffreys_posterior)[::-1]
sorted_jeffreys_posteriors = jeffreys_posterior[sorted_indices_jeffreys]
sorted_S_values_jeffreys = S_values[sorted_indices_jeffreys]

# Compute 95% credible region for Jeffreys prior
cumulative_prob_jeffreys = np.cumsum(sorted_jeffreys_posteriors)
credible_region_indices_jeffreys = np.where(cumulative_prob_jeffreys <= 0.95)[0]
S_lower_jeffreys = np.min(sorted_S_values_jeffreys[credible_region_indices_jeffreys])
S_upper_jeffreys = np.max(sorted_S_values_jeffreys[credible_region_indices_jeffreys])
most_probable_jeffreys = sorted_S_values_jeffreys[0]

# Repeat for uniform prior
sorted_indices_uniform = np.argsort(uniform_posterior)[::-1]
sorted_uniform_posteriors = uniform_posterior[sorted_indices_uniform]
sorted_S_values_uniform = S_values[sorted_indices_uniform]
cumulative_prob_uniform = np.cumsum(sorted_uniform_posteriors)
credible_region_indices_uniform = np.where(cumulative_prob_uniform <= 0.95)[0]
S_lower_uniform = np.min(sorted_S_values_uniform[credible_region_indices_uniform])
S_upper_uniform = np.max(sorted_S_values_uniform[credible_region_indices_uniform])
most_probable_uniform = sorted_S_values_uniform[0]

# Display the results in a table
credible_regions = {
    "Jeffreys Prior": {
        "Most Probable Signal Strength": most_probable_jeffreys,
        "Lower Bound (95% Credible)": S_lower_jeffreys,
        "Upper Bound (95% Credible)": S_upper_jeffreys
    },
    "Uniform Prior": {
        "Most Probable Signal Strength": most_probable_uniform,
        "Lower Bound (95% Credible)": S_lower_uniform,
        "Upper Bound (95% Credible)": S_upper_uniform
    }
}

credible_regions_df = pd.DataFrame(credible_regions)
print(credible_regions_df)

# Step 7: Including uncertainty in the line frequency

# Likelihood function with uncertain line center frequency
def likelihood_with_uncertainty(S, freq_center, data, sigma_L, sigma_noise):
    model = gaussian_spectral_line(S, np.arange(1, 65), freq_center, sigma_L)
    return np.exp(-0.5 * np.sum(((data - model) / sigma_noise) ** 2))

# Frequency channel range (from 1 to 50)
frequency_range = np.arange(1, 51)

# Initialize grid for the posterior
likelihood_grid = np.zeros((len(S_values), len(frequency_range)))

# Compute likelihoods for all signal strengths and frequencies
for i, S in enumerate(S_values):
    for j, freq in enumerate(frequency_range):
        likelihood_grid[i, j] = likelihood_with_uncertainty(S, freq, data, sigma_L, sigma_noise)

# Normalize to represent posterior probabilities
posterior_grid = likelihood_grid / np.sum(likelihood_grid)

# Plotting the 2D posterior distribution (signal strength vs. frequency)
plt.figure(figsize=(12, 8))
plt.contourf(frequency_range, S_values, posterior_grid, cmap='viridis')
plt.colorbar(label='Posterior Probability')
plt.title('Posterior Distribution for Signal Strength and Frequency (Uniform Priors)')
plt.xlabel('Line Center Frequency (Channel)')
plt.ylabel('Signal Strength')

