# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:02:56 2024

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import math 
from mpl_toolkits.mplot3d import Axes3D


#known values
A1 = 4.802033
A2 = 4.43181
x1, y1 = 0.5, 0.5
x2, y2 = 0.65, 0.75
sigma1, sigma2 = 0.2, 0.04


#define the PDF for X and Y
def posterior_joint(X, Y):
    xterm = A1 * np.exp(-((X - x1)**2 + (Y - y1)**2) / (2 * sigma1**2))
    yterm = A2 * np.exp(-((X - x2)**2 + (Y - y2)**2) / (2 * sigma2**2))
    return xterm + yterm



# Define the grid for X and Y in the interval [0, 1]
x_vals = np.linspace(0, 1, 100)
y_vals = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_vals, y_vals)

#calculate the PDF
posterior = posterior_joint(X,Y)
# print(posterior)

#plots for part a and b

# a) Contour plot in the interval [0, 1] for x and y
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, posterior, 20, cmap='viridis')
plt.title('Contour Plot of the Joint Probability Density Function')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.colorbar(contour)
plt.show()

# b) 3D plot of the distribution
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, posterior, cmap='viridis', edgecolor='none')
ax.set_title('3D Plot of the Joint Probability Density Function')
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_zlabel('p(X, Y | D, M, I)')
plt.show()


#part c

marginal_X = np.trapz(posterior, y_vals, axis=0)

# Marginal for Y (integrating over X)
marginal_Y = np.trapz(posterior, x_vals, axis=1)

# Calculate the integrals of the marginals (before normalization)
integral_X_before = np.trapz(marginal_X, x_vals)
print(integral_X_before)
integral_Y_before = np.trapz(marginal_Y, y_vals)
print(integral_Y_before)
# Normalize the marginals
marginal_X /= integral_X_before
marginal_Y /= integral_Y_before

# Calculate the integrals after normalization to check if they are equal to 1
integral_X_after = np.trapz(marginal_X, x_vals)
integral_Y_after = np.trapz(marginal_Y, y_vals)

print("the x integral is", integral_X_before, "so it isn't = 1. This means we need to normalize the marginal probability.")


#plot comparison of the marginal PDFs
plt.figure(figsize=(8,6))
plt.plot(x_vals, marginal_X, label = 'p(X|D,M,I)', color = 'red')
plt.plot(y_vals, marginal_Y, label = 'p(Y|D,M,I)', color = 'blue')
plt.xlabel("X/Y")
plt.ylabel("Probability density")
plt.legend()
plt.show()



